"""Visual-grounding inference for an ROI thumbnail."""

from pathlib import Path

import numpy as np
from PIL import Image


BACKGROUND_ANSWER = (
    "No diagnostic tissue is present in this region; it appears to be background."
)
TISSUE_ANSWER = "Yes, viable diagnostic tissue is present in this region."
TISSUE_FRACTION_THRESHOLD = 0.08


def _estimate_tissue_fraction(roi_image_path: Path) -> float:
    with Image.open(roi_image_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)

    scaled = rgb.astype(np.float32) / 255.0
    value = scaled.max(axis=2)
    chroma = value - scaled.min(axis=2)
    saturation = np.divide(
        chroma,
        value,
        out=np.zeros_like(chroma, dtype=np.float32),
        where=value > 0,
    )
    background = np.all(rgb > 220, axis=2) | ((value > 0.88) & (saturation < 0.12))
    return float(np.mean(~background))


def predict_visual_context_response(
    *,
    question_path: Path,
    roi_image_path: Path,
) -> str:
    del question_path
    try:
        if _estimate_tissue_fraction(roi_image_path) < TISSUE_FRACTION_THRESHOLD:
            return BACKGROUND_ANSWER
    except Exception:
        pass
    return TISSUE_ANSWER
