# Third-Party Notices

This file records the principal external models, implementations, and runtime
software used by REG2026 v0.6.0. It is not a substitute for the complete license
texts included by package distributions and the container base image.

## UNI2-h

- Project: MahmoodLab UNI2-h
- Source: <https://huggingface.co/MahmoodLab/UNI2-h>
- Paper: Chen et al., *Towards a General-Purpose Foundation Model for
  Computational Pathology*, Nature Medicine (2024),
  <https://doi.org/10.1038/s41591-024-02857-3>
- License: CC BY-NC-ND 4.0 with gated access and additional stated use terms

UNI2-h permits non-commercial academic research use with attribution. Access
requires individual registration and acceptance of the upstream terms. The
upstream terms prohibit redistribution of the checkpoint and restrict commercial
use and derivatives. The checkpoint in this release must be supplied only
through the authorized review channel.

## ACMIL

- Project: Attention-Challenging Multiple Instance Learning
- Source: <https://github.com/dazhangyu123/ACMIL>
- License: MIT
- Paper: Zhang et al., *Attention-Challenging Multiple Instance Learning for
  Whole Slide Image Classification*, ECCV 2024

The diagnosis heads use a local gated-attention MIL implementation informed by
the published ACMIL architecture.

## Runtime software

| Component | License |
|---|---|
| PyTorch and TorchVision | BSD 3-Clause |
| timm | Apache License 2.0 |
| NumPy | BSD 3-Clause |
| Pillow | HPND |
| tifffile | BSD 3-Clause |
| imagecodecs | BSD 3-Clause |
| Zarr | MIT |
| NVIDIA CUDA and cuDNN components in the base image | NVIDIA software license terms |

Transitive Python packages are installed by `pip` as dependencies of the
versions pinned in `requirements.txt`. Their metadata and license files remain
available in the Python environment.

## Challenge-derived artifacts

`slot_medoids.json`, `exemplar_bank.npz`, and `exemplar_cots.json` are derived
from the REG2026 challenge training release. They are restricted to authorized
challenge review and evaluation and must not be published, redistributed, or
used outside the applicable challenge terms.
