# Model Card: CYX-AI REG2026 v0.6.0

## Model details

| Field | Value |
|---|---|
| Team | CYX-AI |
| Primary contributor | Yixin Chen |
| Version | 0.6.0 |
| Domain | Histopathology whole-slide and ROI analysis |
| Interfaces | Visual grounding; workflow reasoning |
| Backbone | Frozen UNI2-h ViT-H/14, 1536-dimensional tile embeddings |
| Downstream models | 25-head organ router; per-organ multi-scale gated-attention MIL diagnosis ensembles |
| Output method | Diagnosis-conditioned nearest-exemplar retrieval with medoid fallbacks |

## Intended use

The model is intended only for authorized REG2026 challenge review and
evaluation. It is not a medical device, does not provide a clinical diagnosis,
and must not be used for patient care or other operational clinical decisions.

The controlled assets and derived outputs must not be redistributed. UNI2-h use
is additionally limited by its gated CC BY-NC-ND 4.0 terms.

## Inputs and outputs

Interface 0 accepts a visual-context question and an ROI JPEG. It returns a
single JSON string describing whether diagnostic tissue is present.

Interface 1 accepts one TIFF whole-slide image. It returns a JSON array of
workflow steps with `question`, `answer`, and `next_question` fields.

## Architecture and inference

Interface 1 samples tissue-rich regions at two scales and encodes tiles with
frozen UNI2-h. A center-balanced linear-head ensemble predicts organ. Per-organ
gated-attention MIL ensembles predict the primary diagnosis from denser
multi-scale bags on CUDA. The normalized mean slide embedding retrieves the
nearest exemplar within the predicted organ/diagnosis group.

Fallback order is diagnosis medoid, organ medoid, then global medoid. Process
isolation, a fixed inference deadline, and schema validation provide bounded and
valid output behavior.

## Training and data

The downstream heads were trained from the controlled REG2026 training release.
The organ router uses center-balanced resampling over frozen multi-scale tile
features. Diagnosis models use full-tissue feature bags, per-organ label
vocabularies, multi-seed bagging, and center balancing. The exemplar and medoid
tables are derived from challenge training workflows.

The foundation-model checkpoint was not trained by CYX-AI. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution and license
terms.

## Model assets and hashes

The checked-in release lock is `configs/artifacts-v0.6.0.json`.
`model/MANIFEST.sha256` must match it exactly and is verified both before and
during image construction.

| Asset | Purpose | SHA-256 in this checkout |
|---|---|---|
| `uni2h/pytorch_model.bin` | Frozen UNI2-h backbone | `6e077eda234bebc595868d918d3458d9dd32a050199b0ff04443b2f46a0a3b1e` |
| `organ_uni2h_ms_ensemble.pt` | 25-head organ router | `e2c6bc3617906a6442934c5e6cc795ec7f7d9bb47ea8af0bfdaa4c05e38294e8` |
| `organ_dx_ensemble.pt` | Per-organ diagnosis ensembles | `030bfd436bc3b84c729b14f19597979971846ff67e9c30455eaa86d231b023af` |
| `slot_medoids.json` | Diagnosis, organ, and global fallback workflows | `85c1113fbe28982eada3382d8157571117084f6a336227ca346a91f90e876972` |
| `exemplar_bank.npz` | Normalized WSI embeddings and retrieval index | `30cba112e3e2ae1d38fc295e0496c26eb4adacaf633110e8e076855d23e48c77` |
| `exemplar_cots.json` | Challenge workflow exemplars | `56c621a27b5508aa56e820818abdbd2fe35ef8a70abe271f650c7b501efb8363` |

These are the selected v0.6.0 artifact identifiers. The release is accepted only
when the file set and all six hashes match the checked-in lock.

## Limitations and risks

- The supported label space is limited to seven organs and diagnoses represented
  in REG2026.
- Performance may degrade under scanner, stain, preparation, population, or
  disease shifts.
- Sparse sampling can miss small lesions or rare morphology.
- Diagnosis inference requires CUDA; CPU fallback omits the diagnosis heads.
- Retrieval returns an existing challenge workflow and may not reflect every
  finding in the input slide.
- Medoid fallback outputs are valid but can be nonspecific.
- Interface 0 is a tissue-presence detector rather than a general visual
  question-answering model.
- No prospective clinical validation, calibration for clinical risk, or fairness
  assessment has been performed.

## Reproducibility

Build, test, release, and training commands are documented in
[README.md](README.md). Exact reproduction also requires the controlled
challenge release, approved UNI2-h access, and the reviewed asset manifest.
