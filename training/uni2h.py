from __future__ import annotations

from pathlib import Path

import numpy as np

from training.common import verify_sha256

MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
FEATURE_DIMENSION = 1536


def load_model(weights: Path, expected_sha256: str | None, device: str):
    import timm
    import torch
    from timm.layers import SwiGLUPacked

    verify_sha256(weights, expected_sha256)
    kwargs = {
        "img_size": 224,
        "patch_size": 14,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embed_dim": FEATURE_DIMENSION,
        "mlp_ratio": 2.66667 * 2,
        "num_classes": 0,
        "no_embed_class": True,
        "mlp_layer": SwiGLUPacked,
        "act_layer": torch.nn.SiLU,
        "reg_tokens": 8,
        "dynamic_img_size": True,
    }
    model = timm.create_model("vit_giant_patch14_224", pretrained=False, **kwargs)
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint, strict=True)
    return model.eval().to(device)


def encode_tiles(model, tiles: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    import torch

    features = []
    with torch.inference_mode():
        for start in range(0, len(tiles), batch_size):
            batch = tiles[start : start + batch_size].astype(np.float32) / 255.0
            batch = (batch - MEAN) / STD
            tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=device.startswith("cuda"),
            ):
                output = model(tensor)
            features.append(output.float().cpu().numpy())
    if not features:
        return np.zeros((0, FEATURE_DIMENSION), dtype=np.float32)
    return np.concatenate(features).astype(np.float32)
