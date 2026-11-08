# Training recipe (tier 2)

> **Status: runnable.** These are the commands that produced the weights; `python scripts/train.py full` runs them
> in order as resumable steps (`src/filament_seg/training.py`, in its own run root, never over the shipped weights),
> and the notebook's section 2 runs them with `TRAIN = True`. Verified: `scripts/train.py smoke-full` runs the same
> commands end to end on 10 training cases (5 folds x 1 epoch, both networks, every out-of-fold step); the selector
> fit, run with `selector_stage fit --retrain` on the out-of-fold maps of the shipped networks, reproduces the shipped
> bag (identical cut and predictions). The 250-epoch trainings themselves were not re-run. Prediction uses D007 and D036 fold 0, whose weights ship in `weights/fast/`; folds 1–4 of the same two
> networks are needed for the selector's out-of-fold rows. Two warnings:
> - **Not bit-reproducible.** nnU-Net training on GPU is not deterministic, so a retrain gives
>   different weights and a submission that differs slightly. Training-time environment: torch
>   2.13.0+cu130, cuDNN 9.20.0, one RTX PRO 3000 Blackwell 12 GB (nnU-Net `debug.json`).
> - **Labels.** `scripts/build_labels.py` builds the networks' labels from the competition training COCO
>   JSON. They equal the labels the released weights were trained on, value for value, on 707 / 707 images
>   (`tests/eval_records/build_labels_record.json`).

## 0. Data

```bash
python scripts/prepare_data.py --channels --split both        # channels of the 707 + 180 images (prep/channels.py)
python scripts/prepare_data.py --selector-tables              # selector tables (competition COCO + case mapping)
```

```bash
python scripts/build_labels.py                                # labelsTr of D005/D007/D012/D036 (prep/labels.py)
```

Labels (`filament_seg.prep.labels`; source: `MAGFiLO_1.0_Annotations_kaggle2026_train.json`, nothing else):
- Dataset005 / 007 / 012: one union label per image (uint8, 0/1), the OR over every annotator record of the
  pycocotools rasterisation (`frPyObjects` + `merge` + `decode`) of each polygon.
- Dataset036: three classes (0/1/2), with filament barbs as class 2. The barbs are geometric: every skeleton pixel
  off the geodesic-diameter path of an object of at least 200 px, dilated to the object's median half-width
  (`barb_label` in `prep/labels.py`).
- Cases `img_XXXX` number the training JPEGs in sorted file-name order (checked against
  `configs/case_mapping_full.json`).

**Splits:** identical for every dataset. Sort the 707 case ids and take `folds = [cases[i::5] for i in range(5)]`;
`splits_final.json[k] = {"train": all cases not in fold k, "val": sorted(fold k)}`. Write the file into
`nnUNet_preprocessed/<Dataset>/` after preprocessing (`python scripts/train.py splits --dataset <Dataset>`); it
reproduces the shipped splits exactly (`tests/test_training.py`).

## 1. nnU-Net networks (stock nnU-Net v2.8.1, stock trainer, 2d, default plans)

For each dataset, D007 (`-d 7`) and D036 (`-d 36`):

```bash
nnUNetv2_plan_and_preprocess -d <id> -c 2d --verify_dataset_integrity
# write splits_final.json (above)
for F in 0 1 2 3 4; do nnUNetv2_train <id> 2d $F -tr nnUNetTrainer_250epochs; done
```

- Epochs: 250 (`nnUNetTrainer_250epochs`, stock). Folds 0–4. No seed is set; nnU-Net is not deterministic.
- Time: about 6.5 h per fold on the laptop GPU (D007 fold timestamps 2026-07-24…26). That is 10 fold trainings in
  all.
- Environment: `requirements/nnunet-gpu.lock.txt`. `nnUNet_n_proc_DA=12` was set.

## 2. Out-of-fold predictions (the selector's training inputs)

| artefact (`$FILAMENT_WORK/predictions/…`) | command |
|---|---|
| `vessel_oof_probs` (D007, u16 maps) | `predict_oof_probs.py --out … --step-size 0.75 --folds 0 1 2 3 4` (defaults: D007, `--fg class1`) |
| `oof_d036` | `make_oof_set.py --dataset Dataset036_LimbDarkBarbClass --from-dirs …/fold_%d/validation_binary`, where `validation_binary` = `merge_barb_predictions.py` (label > 0) |

Every map is the single model of the fold that held the image out. Map writer:
`(clip(p, 0, 1) * 65535).astype(uint16)` (truncation). The selector's second member, P, is not a file: the selector
stage computes it from `vessel_oof_probs` (p ≥ 0.5), the same map its candidates come from.

## 3. Selector

```bash
python -m filament_seg.selector_stage fit               # ~5.5 min CPU (the shipped weights/selector/FAST_bag.pkl)
```

The evaluation does not use this bag: `scripts/eval_oof_selection.py` cross-fits the selector on the out-of-fold
rows (`REPRODUCE.md`, Evaluation).
