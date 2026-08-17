# REG2026 v0.6.0

Reference implementation for the CYX-AI REG2026 submission.

- Team: CYX-AI
- Team member: Yixin Chen
- Grand Challenge: [CYX-AI](https://grand-challenge.org/users/CYX-AI/)
- GitHub: [YixinChen-AI](https://github.com/YixinChen-AI)
- Container target: Grand Challenge, `linux/amd64`
- Interfaces: visual grounding and workflow reasoning
- Runtime: offline, one case per container invocation

This repository contains inference, packaging, and reproduction code. Large
checkpoints and controlled challenge data are not included. Challenge-derived
retrieval artifacts are excluded from Git and remain subject to the applicable
challenge terms.

The submitted system is version `0.6.0`. This repository preserves its
inference path and documents deterministic training, packaging, and integrity
checks. Required model assets are identified by SHA-256 hashes in
`configs/artifacts-v0.6.0.json`.

## Method

### Interface 0: visual grounding

The visual-context question and ROI thumbnail are read from the fixed Grand
Challenge sockets. A deterministic tissue-presence detector estimates the
non-background fraction of the thumbnail and returns a JSON string response.

### Interface 1: workflow reasoning

The whole-slide pipeline:

1. samples tissue-rich tiles at 256 px and 512 px;
2. encodes resized 224 px tiles with the frozen UNI2-h ViT-H/14 backbone;
3. predicts one of seven organs with a 25-head, center-organ weighted linear
   ensemble;
4. predicts the primary diagnosis with per-organ, multi-scale gated-attention
   MIL ensembles when a CUDA device is available;
5. retrieves the nearest challenge-training exemplar within the predicted
   organ/diagnosis group; and
6. falls back to diagnosis-, organ-, or global-medoid workflows when a more
   specific result is unavailable.

Inference runs in an isolated process with a fixed time budget. Output validation
and deterministic fallbacks prevent malformed results from reaching the output
socket.

## Interface contract

| Interface | Inputs | Output |
|---|---|---|
| Visual grounding | `/input/visual-context-question.json`, `/input/histopathology-region-of-interest-thumbnail.jpeg` | `/output/visual-context-response.json` containing a JSON string |
| Workflow reasoning | `/input/images/whole-slide-image/<uid>.tiff` | `/output/chain-of-thought.json` containing a JSON array |

`/input/inputs.json` selects the interface. Each workflow step has non-empty
`question` and `answer` strings and a `next_question` string. The final
`next_question` is empty.

The entrypoint is:

```text
python inference.py
```

## Model staging

Before building, obtain the required assets from their licensed sources and
place them under `model/`:

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

`configs/artifacts-v0.6.0.json` is the checked-in release lock.
`MANIFEST.sha256` uses the standard `sha256sum` format with paths relative to
`model/` and must contain exactly the same six paths and hashes. Generate it
only from the corresponding release files:

```bash
cd model
shasum -a 256 \
  uni2h/pytorch_model.bin \
  organ_uni2h_ms_ensemble.pt \
  organ_dx_ensemble.pt \
  slot_medoids.json \
  exemplar_bank.npz \
  exemplar_cots.json > MANIFEST.sha256
```

`do_build.sh` verifies the exact file set, manifest, and all six hashes before
Docker starts. The Dockerfile repeats that verification after copying the local
staging directory. Extra files and duplicate manifest entries are rejected. The
build does not download model weights.

Asset purposes and checksums are recorded in [MODEL_CARD.md](MODEL_CARD.md).
UNI2-h access and use remain subject to its gated license terms. The exemplar
files contain challenge-derived content and are not redistributed here.

## Build and test

Prerequisites are Docker with BuildKit support and, for full Interface 1
inference, an NVIDIA GPU exposed to Docker.

```bash
./do_build.sh
./do_test_run.sh
```

The local test uses fixtures under `test/input/interf0` and
`test/input/interf1`, runs the image with networking disabled, and validates the
JSON output shape. A test WSI must be staged at
`test/input/interf1/images/whole-slide-image/<uid>.tiff`; its name must match the
workflow entry in `test/input/interf1/inputs.json`.

To test one interface:

```bash
INTERFACES=interf0 ./do_test_run.sh
INTERFACES=interf1 ./do_test_run.sh
```

Interface 1 testing fails when Docker GPU access is unavailable. Set
`ALLOW_CPU_FALLBACK=1` only when validating the documented fallback path.

Export the container image:

```bash
./do_save.sh
```

This creates `reg2026_v0.6.0_amd64.tar.gz`. Model assets are already verified and
embedded in the image; no separate model archive is produced.

## Training reproduction

Training requires access to the REG2026 training release and UNI2-h. Install
the training dependencies and verify the checked-in diagnosis plan:

```bash
python -m pip install --requirement requirements-training.txt
python -m training.generate_dx_config \
  --out configs/dx-heads-v0.6.0.json \
  --check
```

Run the complete deterministic recipe from the repository root:

```bash
python -m training.run_recipe \
  --recipe configs/recipe-v0.6.0.json \
  --set WSI_ROOT=data/wsis \
  --set TRAIN_COT=data/train_CoT.json \
  --set TRAIN_COT_SHA256=<sha256> \
  --set UNI2H_WEIGHTS=assets/uni2h/pytorch_model.bin \
  --set UNI2H_SHA256=6e077eda234bebc595868d918d3458d9dd32a050199b0ff04443b2f46a0a3b1e
```

The ordered stages build the case manifest, extract sparse and full-tissue
multi-scale UNI2-h features, train the organ and diagnosis ensembles, build the
medoid and exemplar retrieval assets, and assemble a verified artifact set.
Expensive stages are resumable.

The diagnosis plan contains 203 deployed `K=1` models. Each of the seven organ
routes uses seeds 1 through 20 at 256 px and seeds 1 through 9 at 512 px, so a
single case evaluates 29 diagnosis models after organ routing. Every diagnosis
model uses 40 epochs, `center=ALL`, and `subsample=1.0`. The organ router
contains 25 linear heads trained for three epochs.

Independently verify the assembled release:

```bash
python -m training.artifacts verify --root artifacts/v0.6.0
```

The assembly output keeps all six selected assets under `model/` and writes
`model/MANIFEST.sha256` for the Docker build.

## Repository scope

| Path | Purpose |
|---|---|
| `inference.py`, `core.py` | dispatch, fixed socket paths, output handling |
| `src/interf0/` | visual-grounding implementation |
| `src/interf1/` | WSI classification, retrieval, and fallback logic |
| `training/`, `configs/` | deterministic v0.6.0 reproduction pipeline |
| `model/` | local model-asset staging; large files are ignored by Git |
| `Dockerfile`, `requirements.txt` | offline runtime image |
| `do_build.sh`, `do_test_run.sh`, `do_save.sh` | packaging commands |
| `MODEL_CARD.md` | model details, hashes, intended use, limitations |
| `THIRD_PARTY_NOTICES.md` | external models, software, and licenses |

## Known limitations

- The system is designed for the organs, labels, staining patterns, scanners,
  and workflow language represented in REG2026.
- Interface 0 detects tissue presence; it does not perform open-ended
  question answering.
- Diagnosis heads run only on CUDA. CPU execution falls back to organ-level
  retrieval and is not equivalent to the submitted GPU path.
- Sparse tile sampling can miss small or spatially isolated findings.
- Retrieval reproduces a training workflow rather than generating a new
  case-specific explanation.
- Runtime failures and timeouts return medoid fallbacks, which preserve schema
  validity but may be clinically nonspecific.
- Outputs are for challenge evaluation only and are not for clinical use.

See [MODEL_CARD.md](MODEL_CARD.md) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before running the model.
