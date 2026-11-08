# Package map (`src/filament_seg/`)

**CPU stage**, from the network outputs to the CSV. It is checked by the md5 smoke test (`tests/smoke/run_cpu_smoke.py`). Settings come
only from `configs/<slot>.json` through `config.SlotConfig`. No module changes the working directory or holds
module-level state.

| module | role |
|---|---|
| `config` | `SlotConfig` (frozen dataclass), built from `configs/<slot>.json`; every path is resolved under `FILAMENT_WORK` |
| `maps` | uint16 map I/O, the truncation quantiser, the nnU-Net npz foreground |
| `candidates` | the base mask (T97), >= 300 px and 100–299 px component labelling |
| `selector_features` | member masks (file member, or the base map >= t), per-object features, FLOOR rows, the 22-column design matrix (k = 2 members) |
| `scorer` | per-record targets and host-notebook PQ accounting (`Ref`); COCO grouping by record |
| `selector_fit` | out-of-fold rows, the row table, the 5-member HGB bag and the cut |
| `render` | test-side rows, keep rule, kept-component PNGs, CSV and verify |
| `submission.rle`, `submission.verify` | COCO-RLE writer (LF, non-overlapping rows) and the decode-and-compare gate |
| `selector_tables` | case list plus per-record COCO counts, from competition data only |
| `selector_stage` | CLI: fit / apply / both (`--retrain`: report, do not gate, the cut and CSV of retrained members) |
| `training` | the training recipe as resumable steps (full, smoke, smoke-full), the splits rule, its own run root |

**Inputs to the CPU stage.** These are outside the md5 gate.

| module | role |
|---|---|
| `prep.channels`, `prep.prepare` | network input channels from the raw competition JPEGs (checked array-equal to the stored ones) |
| `gpu.pipeline`, `gpu.nnunet_binaries` | the `--from-images` plan of Ours-Fast: resumable steps, interpreters, weights layout |
| `gpu/predict_oof_probs.py`, `gpu/make_oof_set.py`, `gpu/merge_barb_predictions.py` | the nnU-Net out-of-fold map predictor, out-of-fold assembly |

**Offline evaluation** (`evaluation/`, paper § Evaluation). Outside the md5 gate: it reads the CPU stage's out-of-fold
rows and writes no submission. Expected values: `configs/eval/expected.json`; inputs: `configs/eval/inputs.json`.

| module | role |
|---|---|
| `evaluation.oof_selection` | cross-fitted selector keeps and scores on the 707 OOF rows; kept OOF masks |
| `evaluation.host_rows` | per-record host rows from masks (TP pairs, FP, FN, m2n, pair IoU); numpy torchmetrics-equivalent Dice |
| `evaluation.bootstrap` | pooled ratio-of-sums and mean-paired-difference bootstraps on shared image draws |
| `evaluation.heldout_annotator` | held-out annotator record vs FAST / D007 (Table 1) |
| `evaluation.calibration` | ECE of the selector score, cross-fitted isotonic, test score scale |
| `evaluation.morphometry` | width, branching, split degree, barb coverage |
| `evaluation.bundle` | the out-of-fold evaluation bundle (`eval_bundle/`): export from the full maps, manifest check, staging into WORK |
| `evaluation.expected` | the gate: published values per model set, comparison helpers |
