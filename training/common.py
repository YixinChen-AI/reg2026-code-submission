from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ORGANS = ("bladder", "breast", "cervix", "colon", "lung", "prostate", "stomach")
DX_QUESTION = "what is the #1 diagnosis"
GROUP_SEPARATOR = "|||"
ALIASES = {
    "pridominant": "predominant",
    "dianoses": "diagnoses",
    "diagnosises": "diagnoses",
    "includes": "include",
}
TERMINAL_TOKENS = {
    "",
    "end",
    "stop",
    "finish",
    "finished",
    "none",
    "null",
    "no next question",
    "no further question",
}


def canonicalize_text(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").lower()).strip()
    text = re.sub(r"[\s\.,;:!?]+$", "", text).strip()
    for source, target in ALIASES.items():
        text = re.sub(rf"\b{re.escape(source)}\b", target, text)
    return text


def canonicalize_next_question(value: Any) -> str:
    text = canonicalize_text(value)
    return "__END__" if text in TERMINAL_TOKENS else text


def load_cases(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("cases", "data"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError(f"unsupported case JSON structure: {path}")


def diagnosis_of(case: dict[str, Any]) -> str | None:
    for step in case.get("chain-of-thought", []):
        if canonicalize_text(step.get("question")) == DX_QUESTION:
            value = canonicalize_text(step.get("answer"))
            return value or None
    return None


def wsi_stem(value: Any) -> str:
    name = Path(str(value)).name
    lower = name.lower()
    if lower.endswith(".tiff"):
        return name[:-5]
    if lower.endswith(".tif"):
        return name[:-4]
    return name


def center_of(value: Any) -> str:
    return "_".join(wsi_stem(value).split("_")[:2])


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: Path, expected: str | None) -> str:
    actual = sha256_file(path)
    if expected and actual.lower() != expected.lower():
        raise ValueError(
            f"SHA-256 mismatch for {path}: expected {expected}, got {actual}"
        )
    return actual


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(value)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".npy", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        np.save(stream, value)
    try:
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_npz(path: Path, **values: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        np.savez_compressed(stream, **values)
    try:
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_torch_save(path: Path, value: Any) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def ordered_files(directory: Path, suffix: str) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.name.endswith(suffix))


def read_nonempty_lines(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def require_nonempty(items: Iterable[Any], name: str) -> list[Any]:
    result = list(items)
    if not result:
        raise ValueError(f"{name} is empty")
    return result
