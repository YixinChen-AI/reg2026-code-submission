# REG2026 v0.6.0 reproduction

This directory reproduces the v0.6.0 model assets from WSIs, `train_CoT.json`,
and the UNI2-h state dict. Generated data, features, checkpoints, logs, and
artifacts are intentionally excluded from the repository.

Run commands from the repository root. Install `requirements-training.txt`,
then generate or verify the checked-in diagnosis plan:

```bash
python -m training.generate_dx_config --out configs/dx-heads-v0.6.0.json --check
```

The diagnosis plan contains exactly 203 deployed ABMIL checkpoints: seven
organs, 20 seeds at 256 px and 9 seeds at 512 px. Every checkpoint uses `K=1`,
40 epochs, `center=ALL`, and `subsample=1.0`. Its deterministic first 10% set
overlaps the full training set and is used only for checkpoint selection; it is
not out-of-bag or held-out validation.

Execute the machine-readable recipe on allocated compute:

```bash
python -m training.run_recipe \
  --recipe configs/recipe-v0.6.0.json \
  --set WSI_ROOT=/path/to/wsis \
  --set TRAIN_COT=/path/to/train_CoT.json \
  --set TRAIN_COT_SHA256=<sha256> \
  --set UNI2H_WEIGHTS=/path/to/pytorch_model.bin \
  --set UNI2H_SHA256=<sha256>
```

All stages use deterministic ordering. Expensive preprocessing writes atomic
per-WSI shards and skips completed outputs. The diagnosis runner processes the
plan as one resumable loop; `--shard-index` and `--shard-count` are available
when a small fixed number of allocated GPU workers is required.

The final artifact set contains all runtime assets under `model/`,
`model/MANIFEST.sha256` for container construction, and an
`artifact-manifest.json` with byte sizes and SHA-256 digests. Verify it with:

```bash
python -m training.artifacts verify --root artifacts/v0.6.0
```
