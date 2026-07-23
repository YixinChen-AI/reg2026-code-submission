# Controlled model assets

This directory is the local staging area for the reviewed REG2026 v0.6.0
release. Model files are intentionally excluded from Git and must be supplied
through the authorized review channel.

Required layout:

```text
model/
├── MANIFEST.sha256
├── exemplar_bank.npz
├── exemplar_cots.json
├── organ_dx_ensemble.pt
├── organ_uni2h_ms_ensemble.pt
├── slot_medoids.json
└── uni2h/
    └── pytorch_model.bin
```

`MANIFEST.sha256` must list exactly the six assets using paths relative to this
directory and match `configs/artifacts-v0.6.0.json`. Both `do_build.sh` and the
Dockerfile reject missing, duplicate, modified, or additional model files.

Do not obtain UNI2-h from an unofficial mirror or redistribute it. Each reviewer
must have approved access and comply with the upstream license and access terms.
The exemplar bank, exemplar workflows, and medoid workflows are derived from
REG2026 challenge data and are restricted to authorized challenge review and
evaluation.
