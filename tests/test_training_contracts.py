from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from training.artifacts import ARTIFACTS, assemble, verify
from training.build_manifest import build_manifest
from training.build_medoids import build_table
from training.common import ORGANS, GROUP_SEPARATOR, sha256_file, verify_sha256
from training.generate_dx_config import build_config
from scripts.verify_model_assets import verify as verify_model_assets

ROOT = Path(__file__).resolve().parents[1]


def test_checked_in_diagnosis_plan_is_exactly_generated() -> None:
    expected = build_config()
    actual = json.loads(
        (ROOT / "configs/dx-heads-v0.6.0.json").read_text(encoding="utf-8")
    )
    assert actual == expected
    assert actual["head_count"] == 203
    assert actual["model"] == {
        "architecture": "ACMIL_GA",
        "branches": 1,
        "n_masked": 0,
        "mask_drop": 0.0,
        "epochs": 40,
        "center": "ALL",
        "subsample": 1.0,
        "center_balance": False,
        "selection_set": "overlapping_first_10_percent_training_fallback",
    }
    counts = Counter((head["organ"], head["scale"]) for head in actual["heads"])
    assert counts == Counter(
        {
            **{(organ, 256): 20 for organ in ORGANS},
            **{(organ, 512): 9 for organ in ORGANS},
        }
    )
    assert len({head["id"] for head in actual["heads"]}) == 203
    assert len({head["output"] for head in actual["heads"]}) == 203


def test_recipe_covers_complete_chain_and_has_no_machine_paths() -> None:
    recipe_path = ROOT / "configs/recipe-v0.6.0.json"
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    assert {"TRAIN_COT_SHA256", "UNI2H_SHA256"} <= set(recipe["required_variables"])
    modules = [stage["module"] for stage in recipe["stages"]]
    assert modules == [
        "training.build_manifest",
        "training.extract_sparse_tiles",
        "training.extract_sparse_tiles",
        "training.extract_sparse_features",
        "training.tile_full_tissue",
        "training.tile_full_tissue",
        "training.extract_full_tissue_features",
        "training.extract_full_tissue_features",
        "training.train_organ_ensemble",
        "training.train_dx_plan",
        "training.bundle_dx_ensemble",
        "training.build_medoids",
        "training.build_retrieval_bank",
        "training.artifacts",
    ]
    serialized = recipe_path.read_text(encoding="utf-8")
    assert "/share/" not in serialized
    assert "/Volumes/" not in serialized
    assert "/Users/" not in serialized


def test_manifest_is_sorted_relative_and_labelled(tmp_path: Path) -> None:
    wsi_root = tmp_path / "wsis"
    (wsi_root / "nested").mkdir(parents=True)
    (wsi_root / "PIT_02_00002.tiff").touch()
    (wsi_root / "nested/PIT_01_00001.tif").touch()
    (wsi_root / "unmatched.tiff").touch()
    cot = tmp_path / "train.json"
    cot.write_text(
        json.dumps(
            [
                {"id": "PIT_01_00001.tif", "organ": "Colon"},
                {"id": "PIT_02_00002.tiff", "organ": "Lung"},
            ]
        ),
        encoding="utf-8",
    )
    rows, unmatched = build_manifest(cot, wsi_root)
    assert rows == [
        {"path": "PIT_02_00002.tiff", "wsi": "PIT_02_00002", "organ": "lung"},
        {
            "path": "nested/PIT_01_00001.tif",
            "wsi": "PIT_01_00001",
            "organ": "colon",
        },
    ]
    assert unmatched == ["unmatched.tiff"]
    assert all(not Path(row["path"]).is_absolute() for row in rows)


def _case(case_id: str, organ: str, diagnosis: str, middle: str) -> dict:
    return {
        "id": case_id,
        "organ": organ,
        "chain-of-thought": [
            {
                "question": "What is the organ?",
                "answer": organ,
                "next_question": middle,
            },
            {
                "question": middle,
                "answer": "Present",
                "next_question": "What is the #1 diagnosis?",
            },
            {
                "question": "What is the #1 diagnosis?",
                "answer": diagnosis,
                "next_question": "",
            },
        ],
    }


def test_medoid_table_uses_majority_workflow_skeleton() -> None:
    cases = [
        _case("a.tiff", "Colon", "Adenoma", "Is there dysplasia?"),
        _case("b.tiff", "Colon", "Adenoma", "Is there dysplasia?"),
        _case("c.tiff", "Colon", "Adenoma", "Is there invasion?"),
    ]
    table = build_table(cases)
    key = f"colon{GROUP_SEPARATOR}adenoma"
    assert table["organ_dx"][key] == cases[0]["chain-of-thought"]
    assert table["organ"]["colon"] == cases[0]["chain-of-thought"]
    assert table["__global__"] == cases[0]["chain-of-thought"]


def test_sha_verification_rejects_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "value.bin"
    path.write_bytes(b"verified")
    digest = sha256_file(path)
    assert verify_sha256(path, digest) == digest
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_sha256(path, "0" * 64)


def test_artifact_assembly_manifest_detects_tampering(tmp_path: Path) -> None:
    sources = {}
    for index, name in enumerate(ARTIFACTS):
        path = tmp_path / "sources" / f"{name}.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"artifact-{index}".encode())
        sources[name] = path
    output = tmp_path / "release"
    lock = tmp_path / "lock.json"
    lock.write_text(
        json.dumps(
            {
                "release": "0.6.0",
                "files": {
                    ARTIFACTS[name].relative_to("model").as_posix(): sha256_file(path)
                    for name, path in sources.items()
                },
            }
        ),
        encoding="utf-8",
    )
    arguments = {
        **sources,
        "lock": lock,
        "out": output,
        "skip_existing": False,
    }
    assemble(SimpleNamespace(**arguments))
    manifest = verify(output)
    assert len(manifest["files"]) == len(ARTIFACTS)
    checksum_manifest = (output / "model/MANIFEST.sha256").read_text(encoding="utf-8")
    assert "uni2h/pytorch_model.bin" in checksum_manifest
    assert len(checksum_manifest.splitlines()) == len(ARTIFACTS)
    tampered = output / ARTIFACTS["medoids"]
    tampered.write_bytes(b"changed")
    with pytest.raises(ValueError):
        verify(output)


def test_model_staging_rejects_extra_files(tmp_path: Path) -> None:
    root = tmp_path / "model"
    root.mkdir()
    files = {
        "uni2h/pytorch_model.bin": b"backbone",
        "organ_uni2h_ms_ensemble.pt": b"organ",
        "organ_dx_ensemble.pt": b"diagnosis",
        "slot_medoids.json": b"medoids",
        "exemplar_bank.npz": b"bank",
        "exemplar_cots.json": b"workflows",
    }
    hashes = {}
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        hashes[name] = sha256_file(path)
    lock = tmp_path / "lock.json"
    lock.write_text(
        json.dumps({"release": "0.6.0", "files": hashes}),
        encoding="utf-8",
    )
    (root / "MANIFEST.sha256").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in hashes.items()),
        encoding="utf-8",
    )
    verify_model_assets(root, lock)
    (root / "unexpected.bin").write_bytes(b"x")
    with pytest.raises(ValueError, match="unexpected"):
        verify_model_assets(root, lock)
