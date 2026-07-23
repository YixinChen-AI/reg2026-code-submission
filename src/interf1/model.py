"""Diagnosis-conditioned workflow retrieval for whole-slide images."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

MODEL_PATH = Path("/opt/app/model")
if not (MODEL_PATH / "organ_uni2h_ms_ensemble.pt").exists():
    MODEL_PATH = Path("/opt/ml/model")
if not (MODEL_PATH / "organ_uni2h_ms_ensemble.pt").exists():
    MODEL_PATH = Path(__file__).resolve().parents[2] / "model"

ORGANS = ["bladder", "breast", "cervix", "colon", "lung", "prostate", "stomach"]
SCALES = (256, 512)
GRID = 12
K_PER_SCALE = 16
DX_MIN_FRAC = 0.25
DX_MAX_256 = 1200
DX_MAX_512 = 400
DX_GRID_256 = 56
DX_GRID_512 = 36
DX_BUDGET_S = 170.0
ENCODE_RESERVE_S = 35.0
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
GROUP_SEPARATOR = "|||"
_FALLBACK = [
    {
        "question": "What is the final pathology report?",
        "answer": "No diagnostic inference available.",
        "next_question": "",
    }
]
_STATE = None


def _sample_tiles(
    wsi_path,
    tile,
    grid=GRID,
    max_tiles=K_PER_SCALE,
    white_thresh=220,
    min_frac=0.30,
    deadline=None,
):
    """Sample tissue-rich tiles without decoding the complete slide."""
    import tifffile
    import zarr

    store = tifffile.imread(str(wsi_path), aszarr=True)
    try:
        z = zarr.open(store, mode="r")
        if not hasattr(z, "shape"):
            arrs = [z[k] for k in z.keys() if hasattr(z[k], "shape")]
            z = max(arrs, key=lambda a: a.shape[0])
        height, width = z.shape[0], z.shape[1]
        ys = np.linspace(0, max(0, height - tile), grid).astype(int)
        xs = np.linspace(0, max(0, width - tile), grid).astype(int)
        coords = [(int(y), int(x)) for y in ys for x in xs]
        order = range(len(coords))
        if deadline is not None:
            order = np.random.default_rng(0).permutation(len(coords))
        scored = []
        for order_index in order:
            if deadline is not None and time.monotonic() > deadline:
                break
            y, x = coords[int(order_index)]
            patch = np.asarray(z[y : y + tile, x : x + tile])
            if patch.shape[:2] != (tile, tile):
                continue
            not_white = ~np.all(patch[..., :3] > white_thresh, axis=2)
            not_black = ~np.all(patch[..., :3] < 25, axis=2)
            frac = float(np.mean(not_white & not_black))
            if frac >= min_frac:
                scored.append((frac, int(order_index), patch))
    finally:
        store.close()
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [p for _, _, p in scored[:max_tiles]]


def _load():
    global _STATE
    if _STATE is not None:
        return _STATE
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import timm
    import torch
    import torch.nn as nn
    from timm.layers import SwiGLUPacked

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # Architecture specified by the UNI2-h model card.
    encoder_config = dict(
        img_size=224,
        patch_size=14,
        depth=24,
        num_heads=24,
        init_values=1e-5,
        embed_dim=1536,
        mlp_ratio=2.66667 * 2,
        num_classes=0,
        no_embed_class=True,
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU,
        reg_tokens=8,
        dynamic_img_size=True,
    )
    encoder = timm.create_model(
        "vit_giant_patch14_224", pretrained=False, **encoder_config
    )
    encoder.load_state_dict(
        torch.load(MODEL_PATH / "uni2h" / "pytorch_model.bin", map_location="cpu")
    )
    encoder = encoder.to(dev).eval()

    bundle = torch.load(
        MODEL_PATH / "organ_uni2h_ms_ensemble.pt", map_location=dev, weights_only=False
    )
    dim = bundle.get("dim", 1536)
    heads = []
    for st in bundle["states"]:
        head = torch.nn.Linear(dim, 7)
        head.load_state_dict(st)
        head.to(dev).eval()
        heads.append(head)
    mu = np.asarray(bundle["mu"], np.float32)
    sd = np.asarray(bundle["sd"], np.float32)
    organs = bundle.get("organs", ORGANS)

    class GatedAttention(nn.Module):
        def __init__(self, L=768, D=128, K=1):
            super().__init__()
            self.V = nn.Linear(L, D)
            self.U = nn.Linear(L, D)
            self.W = nn.Linear(D, K)

        def forward(self, h):
            a = torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))
            return self.W(a).transpose(0, 1)

    class ACMIL_GA(nn.Module):
        def __init__(self, in_dim=1536, inner=768, K=1, C=2):
            super().__init__()
            self.fc = nn.Sequential(nn.Linear(in_dim, inner), nn.ReLU())
            self.attn = GatedAttention(inner, 128, K)
            # Retained because the submitted checkpoints contain these parameters.
            self.branch_cls = nn.ModuleList([nn.Linear(inner, C) for _ in range(K)])
            self.bag_cls = nn.Linear(inner, C)

        def forward(self, bag):
            h = self.fc(bag)
            attention = torch.softmax(self.attn(h), dim=1)
            pooled = attention.mean(0, keepdim=True) @ h
            return self.bag_cls(pooled)

    dx = {}
    dxp = MODEL_PATH / "organ_dx_ensemble.pt"
    if dxp.exists() and dev == "cuda":
        dxb = torch.load(dxp, map_location=dev, weights_only=False)
        for o, scales in dxb["organs"].items():
            scale_heads = {"e256": [], "e512": []}
            for tag in ("e256", "e512"):
                for checkpoint in scales.get(tag, []):
                    net = ACMIL_GA(
                        1536,
                        768,
                        checkpoint["K"],
                        len(checkpoint["vocab"]),
                    )
                    net.load_state_dict(checkpoint["state"])
                    net.to(dev).eval()
                    scale_heads[tag].append((net, checkpoint["vocab"]))
            dx[o] = scale_heads

    table = json.loads((MODEL_PATH / "slot_medoids.json").read_text(encoding="utf-8"))
    _STATE = {
        "dev": dev,
        "fm": encoder,
        "heads": heads,
        "mu": mu,
        "sd": sd,
        "organs": organs,
        "table": table,
        "torch": torch,
        "dx": dx,
        "bank": _load_bank(),
    }
    return _STATE


def _uni2h_feats(state, tiles):
    """Encode RGB tiles with UNI2-h."""
    torch = state["torch"]
    from PIL import Image

    feats = []
    for i in range(0, len(tiles), 128):
        chunk = tiles[i : i + 128]
        imgs = (
            np.stack(
                [
                    np.asarray(
                        Image.fromarray(t).resize((224, 224), Image.BILINEAR),
                        dtype=np.float32,
                    )
                    for t in chunk
                ]
            )
            / 255.0
        )
        imgs = (imgs - _MEAN) / _STD
        x = torch.from_numpy(imgs).permute(0, 3, 1, 2).float().to(state["dev"])
        with (
            torch.no_grad(),
            torch.autocast(
                "cuda", dtype=torch.float16, enabled=(state["dev"] != "cpu")
            ),
        ):
            f = state["fm"](x)
        feats.append(f.float().cpu().numpy())
    return np.concatenate(feats, 0) if feats else np.zeros((0, 1536), np.float32)


def _predict_organ(state, wsi_path, deadline=None):
    """Predict the organ from a multi-scale tile sample."""
    torch = state["torch"]
    tiles = []
    for ts in SCALES:
        t = _sample_tiles(wsi_path, tile=ts, deadline=deadline)
        if len(t) < 4:
            t = _sample_tiles(wsi_path, tile=ts, grid=GRID + 10, deadline=deadline)
        tiles += t[:K_PER_SCALE]
    if not tiles:
        return None, None
    feats = _uni2h_feats(state, tiles)
    mean_feature = feats.mean(0)
    norm = float(np.linalg.norm(mean_feature))
    emb = (
        (mean_feature / norm).astype(np.float32)
        if norm > 0
        else mean_feature.astype(np.float32)
    )
    standardized = torch.tensor(
        (feats - state["mu"]) / state["sd"],
        dtype=torch.float32,
        device=state["dev"],
    )
    with torch.no_grad():
        probs = sum(
            torch.softmax(head(standardized), 1) for head in state["heads"]
        ) / len(state["heads"])
    probs = probs.mean(0).cpu().numpy()
    return state["organs"][int(probs.argmax())], emb


def _encode_dx(state, tiles, deadline=None):
    """Encode the dense tile bag used by the diagnosis heads."""
    torch = state["torch"]
    from PIL import Image

    feats = []
    for i in range(0, len(tiles), 64):
        if deadline is not None and time.monotonic() > deadline:
            break
        chunk = tiles[i : i + 64]
        imgs = (
            np.stack(
                [
                    np.asarray(
                        Image.fromarray(t).resize((224, 224), Image.BILINEAR),
                        dtype=np.float32,
                    )
                    for t in chunk
                ]
            )
            / 255.0
            - _MEAN
        ) / _STD
        x = torch.from_numpy(imgs).permute(0, 3, 1, 2).float().to(state["dev"])
        with (
            torch.no_grad(),
            torch.autocast(
                "cuda", dtype=torch.float16, enabled=(state["dev"] != "cpu")
            ),
        ):
            f = state["fm"](x)
        feats.append(f.detach().cpu().half().numpy())
    return (
        np.concatenate(feats, 0).astype(np.float32)
        if feats
        else np.zeros((0, 1536), np.float32)
    )


def _predict_dx(state, wsi_path, organ, deadline=None):
    """Predict the primary diagnosis from scale-specific MIL ensembles."""
    torch = state["torch"]
    heads = state["dx"].get(organ)
    if not heads or (not heads["e256"] and not heads["e512"]):
        return None
    sample_deadline = deadline - ENCODE_RESERVE_S if deadline is not None else None
    agg = defaultdict(float)
    scale_configs = (
        (256, heads["e256"], DX_GRID_256, DX_MAX_256),
        (512, heads["e512"], DX_GRID_512, DX_MAX_512),
    )
    for scale, scale_heads, grid, max_tiles in scale_configs:
        if not scale_heads:
            continue
        if deadline is not None and time.monotonic() > deadline:
            break
        try:
            tiles = _sample_tiles(
                wsi_path,
                tile=scale,
                grid=grid,
                max_tiles=max_tiles,
                min_frac=DX_MIN_FRAC,
                deadline=sample_deadline,
            )
            if not tiles:
                continue
            feats = _encode_dx(state, tiles, deadline)
            if len(feats) == 0:
                continue
            bag = torch.tensor(feats, dtype=torch.float32, device=state["dev"])
            with torch.no_grad():
                for net, vocab in scale_heads:
                    p = torch.softmax(net(bag)[0].flatten(), 0).cpu().numpy()
                    for c, pv in zip(vocab, p):
                        agg[c] += float(pv)
        except Exception as e:
            print(f"[interf1] dx scale {scale} failed: {e!r}")
        finally:
            torch.cuda.empty_cache()
    return max(agg, key=agg.get) if agg else None


def _emit_cot(table, organ, dx=None):
    """Return a diagnosis, organ, or global fallback workflow."""
    if dx is not None:
        cot = table.get("organ_dx", {}).get(f"{organ}{GROUP_SEPARATOR}{dx}")
        if cot is not None:
            return cot
    cot = table["organ"].get(str(organ))
    return cot if cot is not None else table["__global__"]


def _load_bank():
    """Load the diagnosis-indexed exemplar bank."""
    try:
        here = Path(__file__).resolve().parent
        bp = here / "exemplar_bank.npz"
        if not bp.exists():
            return None
        with np.load(bp, allow_pickle=False) as bundle:
            embs = bundle["embs"].astype(np.float32)
            wids = bundle["wids"]
            organs = bundle["organs"]
            diagnoses = bundle["dxs"]
        cots = json.loads((here / "exemplar_cots.json").read_text(encoding="utf-8"))
        indices_by_group = defaultdict(list)
        for index, (organ, diagnosis) in enumerate(zip(organs, diagnoses)):
            indices_by_group[(str(organ), str(diagnosis))].append(index)
        return {
            "embs": embs,
            "wids": wids,
            "indices_by_group": {
                key: np.asarray(value) for key, value in indices_by_group.items()
            },
            "cots": cots,
        }
    except Exception as e:
        print(f"[interf1] exemplar bank load failed -> medoid only: {e!r}")
        return None


def _emit_retrieval(state, organ, dx, emb):
    """Retrieve the nearest workflow within an organ-diagnosis group."""
    med = _emit_cot(state["table"], organ, dx)
    bank = state.get("bank")
    if bank is None or emb is None or dx is None:
        return med
    idx = bank["indices_by_group"].get((str(organ), str(dx)))
    if idx is None or len(idx) == 0:
        return med
    sims = bank["embs"][idx] @ emb
    cot = bank["cots"].get(str(bank["wids"][int(idx[int(sims.argmax())])]))
    return cot if cot else med


def predict_chain_of_thought(*, wsi_path: Path):
    """Predict a workflow with deterministic fallback behavior."""
    deadline = time.monotonic() + DX_BUDGET_S
    try:
        state = _load()
        organ, emb = _predict_organ(state, wsi_path, deadline=deadline)
        if organ is None:
            return state["table"]["__global__"]
        dx = None
        try:
            dx = _predict_dx(state, wsi_path, organ, deadline=deadline)
        except Exception as e:
            print(f"[interf1] dx failed -> organ-medoid: {e!r}")
        return _emit_retrieval(state, organ, dx, emb)
    except Exception as e:
        print(f"[interf1] ERROR -> global-medoid fallback: {e!r}")
        try:
            return _load()["table"]["__global__"]
        except Exception:
            return _FALLBACK


def global_fallback_cot():
    """Load the global fallback without initializing the model."""
    try:
        table = json.loads(
            (MODEL_PATH / "slot_medoids.json").read_text(encoding="utf-8")
        )
        g = table.get("__global__")
        return g if isinstance(g, list) and g else _FALLBACK
    except Exception:
        return _FALLBACK
