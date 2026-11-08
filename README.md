# Which Filaments Do Annotators Draw? An nnU-Net with Per-Object Selection for GONG H-α

A Solution to the Solar Filament Segmentation Challenge 2026 (GONG H-alpha).

## Summary

We segment solar filaments in full-disk 2048×2048 GONG H-alpha observations.

- **Ours-Fast (slot `FAST`).** Two networks, both nnU-Nets, fold 0:
  - the **base network**, an nnU-Net on five input channels (c0–c4: raw, CLAHE, background-subtracted, coherence,
    Sato ridge), gives the probability map p. Its 4-connected components of at least 100 px at p ≥ 0.97
    are the candidate objects;
  - a **member network**, an nnU-Net on six channels (c0–c5, adding a limb-darkening-corrected raw; three classes,
    filament and barb merged), gives a binary mask.

  The selector reads two **member masks**: that network's mask, and **P**, the base network's own mask at p ≥ 0.5
  (no extra network: computed from p). Members never add pixels: the selector reads their agreement with each
  candidate as features. A per-object selector keeps a candidate only when it expects the object to raise the
  per-record Panoptic Quality. It is a bag of five gradient-boosted regressors on 22 features (9 of the candidate's
  own probabilities and shape, 8 of member agreement, 5 of image context), trained on the 707 out-of-fold training
  observations. This repository holds the two networks' weights and the selector and predicts with them.

Method, evaluation and figures: the technical report (ACM sigconf, four pages).

## Results

On 296 multi-annotator training observations, out of fold. Each annotator record is held out in turn, and
the model is scored against the remaining records, each record a separate ground truth (the host's
per-record rule). Intervals in the report are 95% paired bootstrap (5000 resamples).

| | Dice | pairwise IoU | PQ (w 0.5) | public LB | file | md5 |
|---|---|---|---|---|---|---|
| held-out annotator | 0.616 | 0.517 | 0.340 | — | — | — |
| **Ours-Fast (FAST)** | 0.673 | 0.591 | 0.444 | 0.39 | `submissions/FAST.csv` | `7c5bd00dbb9165fe28d3d3c83a9b832c` |
| simplified model: the base network alone (p ≥ 0.5, components ≥ 400 px, no selector) | 0.674 | 0.566 | 0.409 | — | — | — |

What each component adds, with intervals, is the paper's ablation table: the selector is the largest step; the second
network adds the least (+0.002 PQ on all 707 out-of-fold observations) for one more network pass.

Dice is `torchmetrics` `DiceScore` with `include_background=False` (the evaluation code implements it in numpy; the
torchmetrics check runs when torchmetrics is installed). The public leaderboard is shown for
identification only: it reports a best-ever score, on about half of the test images, at two decimals. The
offline evaluation is `scripts/eval_*.py` (`REPRODUCE.md`, Evaluation); each script reproduces the reported
values and checks them against `configs/eval/expected.json`.

## Environments

Two Python environments (plus the notebook kernel), with exact pins in `requirements/`:

| file | Python | key packages | used for |
|---|---|---|---|
| `cpu.txt` | 3.12.11 | numpy 2.2.6, scipy 1.18.0, scikit-image 0.26.0, scikit-learn 1.9.0, pillow 12.2.0, pycocotools 2.0.11 | channels, selector, submission |
| `nnunet-gpu.lock.txt` | 3.14.8 | nnunetv2 2.8.1, torch 2.13.0 | the two nnU-Nets |
| `notebook.txt` | 3.12.11 | `cpu.txt` + matplotlib, nbclient, ipykernel | `pipeline.ipynb` |

Reference hardware: NVIDIA RTX PRO 3000 Blackwell Laptop GPU (12 GB), Intel Core Ultra 7 265HX, 31 GB RAM,
Arch Linux. Training stack: torch 2.13.0+cu130, cuDNN 9.20.0.

## What you can run

- **Ours-Fast, entirely from this repository.** Its two networks are stored in `weights/fast/` (0.47 GB, as parts of
  at most 90 MiB because GitHub blocks larger files) and its selector in `weights/selector/`.
  `pipeline.ipynb` runs the whole pipeline with them: optional training (and a short training smoke run), the 180
  test observations from the raw images to the submission and its check against the submitted file, one observation
  in detail, and the evaluation; `python scripts/predict.py --slot FAST --from-images` predicts all 180
  (19.0 min on the reference laptop GPU, "Runtime and memory").
  Both need your competition download and the environments below.
- **The offline evaluation (every out-of-fold number in Results), CPU only, no retraining:** `eval_bundle/` (14 MB,
  tracked) holds the derived out-of-fold inputs; `python -m filament_seg.evaluation.bundle stage` combines it with your
  competition download, then `scripts/eval_*.py` reproduce and gate the reported values (`REPRODUCE.md`, Evaluation).
- **Unit tests, no data needed:** `pip install -e . && pytest` (with `requirements/cpu.txt`; add
  `requirements/eval-test.txt` for the torchmetrics Dice check).

## Quick check (CPU only, about 5 minutes)

Reproduces the paper's Table 1 (our model and a held-out annotator, out of fold) from `eval_bundle/` and your competition
download, with no GPU and no network run; each step exits non-zero if a reported number is not reproduced.

```bash
bash scripts/setup_envs.sh --cpu-only                      # .venv-cpu (Python 3.12), ~1 min
export FILAMENT_KAGGLE_ROOT=/path/to/unzipped/filament-segmentation-2026 FILAMENT_WORK=$PWD/work
.venv-cpu/bin/python -m filament_seg.evaluation.bundle stage --work $FILAMENT_WORK   # ~2.5 min, 2.4 GB in work/
.venv-cpu/bin/python scripts/eval_oof_selection.py         # ~0.5 min
.venv-cpu/bin/python scripts/eval_heldout_annotator.py     # ~1.5 min -> "FAST PQ 0.4443 vs annotator 0.3399 … gate PASS"
.venv-cpu/bin/python -m pytest -q                          # the unit tests, ~1 min
```

`bash scripts/setup_envs.sh` (without `--cpu-only`) also builds the nnU-Net GPU environment for the full prediction below.

## Resources required

| | prediction (180 test images) | whole notebook (+ training smoke run, evaluation) | full retrain |
|---|---|---|---|
| GPU | NVIDIA, CUDA 13.0-capable driver (torch 2.13.0+cu130; reference: driver 615.78, RTX PRO 3000 Blackwell laptop, 12 GB); measured 1.7 GiB GPU memory per process | the smoke training peaked at 5.5 GiB | the same 12 GB GPU was used |
| host RAM | measured 5.8 GiB (summed PSS) | about 11 GiB above idle during the smoke training | not measured; 31 GB machine |
| disk | 6.9 GiB in `$FILAMENT_WORK` | about 11 GiB in `$FILAMENT_WORK` | more (all 887 images' channels, nnU-Net preprocessed data, 10 models) |
| software | Python 3.12 (CPU stage) and Python 3.14 (nnU-Net) environments: 0.6 GB + 5.6 GB installed; the competition download (0.7 GB) | + the notebook kernel (0.8 GB) | |
| time | 19.0 min on the reference laptop (GPU throttled; "Runtime and memory") | 24.5 min | ~65 GPU-h (10 trainings of ~6.5 h) + ~1 h |

Without a CUDA GPU the networks run on the CPU; that path is not tested against the reproduction rule and is far slower.
`python3.12` and `python3.14` can be installed with uv if your system lacks them:
`curl -LsSf https://astral.sh/uv/install.sh | sh` then `uv python install 3.12 3.14` (puts `python3.14` in `~/.local/bin`).

## Quickstart: predict from the stored weights

```bash
git clone https://github.com/frilox042/solar-filament-segmentation-challenge-2026.git && cd solar-filament-segmentation-challenge-2026
# competition data (Kaggle account + accepted rules): https://www.kaggle.com/competitions/filament-segmentation-2026/data
kaggle competitions download -c filament-segmentation-2026 -p ../data && unzip -q ../data/filament-segmentation-2026.zip -d ../data

bash scripts/setup_envs.sh                                   # both environments (or the manual steps below)
```

The same by hand:

```bash
TORCH_INDEX=https://download.pytorch.org/whl/cu130
python3.12 -m venv .venv-cpu    && .venv-cpu/bin/pip install -r requirements/cpu.txt
.venv-cpu/bin/pip install -r requirements/notebook.txt       # for pipeline.ipynb (matplotlib, nbclient, ipykernel)
.venv-cpu/bin/pip install -e .                               # the package (or: export PYTHONPATH=src)
python3.14 -m venv .venv-nnunet && .venv-nnunet/bin/pip install --extra-index-url $TORCH_INDEX -r requirements/nnunet-gpu.lock.txt
```

Point the pipeline at them with absolute paths (each step runs with `$FILAMENT_WORK` as its working directory):

```bash
export FILAMENT_KAGGLE_ROOT=/path/to/unzipped/filament-segmentation-2026   # contains MAGFiLO_1.0_Kaggle_2026/{train,test}/
export FILAMENT_WORK=$PWD/work        # FILAMENT_WEIGHTS defaults to weights/downloaded, where the stored weights are assembled
export FILAMENT_PY_CPU=$PWD/.venv-cpu/bin/python FILAMENT_PY_NNUNET=$PWD/.venv-nnunet/bin/python

python scripts/predict.py --slot FAST --from-images        # -> submissions/out/FAST.csv (assembles weights/fast/ first)
```

Flags: `--dry-run` prints the plan; `--limit 2` is a smoke run on two images (it skips the CPU stage); every step
resumes where it stopped. The CPU stage applies the released selector bag (`weights/selector/FAST_bag.pkl`,
about 2 MB, checked against `weights/MANIFEST.sha256`); `--bag-dir DIR` applies another directory's bag, and
`--refit` re-fits the selector on the 707 out-of-fold observations first.

**CPU stage from cached network outputs:** `python tests/smoke/run_cpu_smoke.py ...` rebuilds the submission file
byte for byte from the saved test network outputs, with negative controls. Those test outputs are not published
(the from-images run regenerates them).

## Full retrain

`python scripts/train.py full` runs the whole recipe below as resumable steps in its own run root
(`$FILAMENT_WORK/train`); `python scripts/train.py smoke-full` runs the same commands on 10 training cases (5 folds x
1 epoch) as a check. `docs/TRAINING.md` lists the commands per network: `nnUNetv2_plan_and_preprocess` and `nnUNetv2_train` for Ours-Fast's
two nnU-Net datasets (Dataset007, the base network, and Dataset036, the member) over folds 0–4; the epochs (250), the folds (`cases[i::5]` over the sorted 707 cases), the
out-of-fold predictions and the selector fit. Fold 0 of each network is the one that predicts the test set; folds 1–4
are needed only for the selector's out-of-fold rows.

Channels come from `python scripts/prepare_data.py --channels`. Training labels come from
`python scripts/build_labels.py`, which rasterises the competition training COCO JSON with pycocotools: the union of
every annotator record, and the barb-class label for Dataset036. These are pixel-identical, on all 707 training
observations, to the labels the shipped nnU-Net networks were trained on. By default the script also writes the union
label of Dataset005 (and Dataset012, the same label; Dataset012 is the single-channel raw-image dataset of the input-channel ablation reported in the paper, not used by Ours-Fast and with no weights shipped), which no network of Ours-Fast trains on: Dataset005's
`labelsTr` is the union reference that `tests/eval_records/records_coco_check.py` compares the COCO records against.
`--datasets D007 D036` builds only the labels that training needs. The channel step of `scripts/predict.py` builds
only the two networks' inputs (Dataset007, Dataset036); prediction reads the test case map in the Dataset007 folder.
Dataset005's directory holds the training case map (`case_mapping_full.json`, `config.TRAIN_CASE_MAP`) that the
selector refit and the evaluation read; it is written by the training-side preparation, not by prediction.

Retraining is not bit-reproducible: nnU-Net training on a GPU is not deterministic.

## Runtime and memory (Ours-Fast, 180 test observations)

Reference run: `scripts/predict.py --slot FAST --from-images` from the raw competition JPEGs and the weights stored
here, on the reference hardware (2026-11-07; per-step wall clock printed by the pipeline; memory sampled every ~6 s from
`/proc` and `nvidia-smi`, so peaks are lower bounds).

| stage | time |
|---|---|
| CPU: input channels of the 180 test images (Dataset007 + Dataset036, 4 images in parallel) | 3.9 min (234 s) |
| GPU: base network, nnU-Net Dataset007 fold 0 (probabilities) | 6.3 min (378 s) |
| GPU: member, nnU-Net Dataset036 fold 0 | 6.7 min (402 s) |
| CPU: selector (member P computed here), CSV, verify and tolerance check | 2.1 min (127 s) |
| **total, end to end** | **19.0 min (1,141 s)** |

**The laptop GPU throttled in this run:** after the first 45–70 s of each network step it ran at about 700 MHz and 54 W
(of a 110 W limit). At full clock a case took 1.2 s, the same as in an earlier run, so on an unthrottled GPU each
network step takes about 3.8 min. Your time depends on your GPU and its cooling.

| memory and disk | |
|---|---|
| host peak (summed PSS of the run's processes) | 5.8 GiB, during the nnU-Net steps (summed RSS, which double-counts shared pages: 7.9 GiB) |
| GPU (per-process `nvidia-smi` used memory) | 1.7 GiB |
| growth over the 180 images | none detected: the per-image PSS trend within each step is consistent with zero (Dataset007 −0.28 [−0.60, +0.33], Dataset036 −0.21 [−0.40, +0.07], channels +0.45 [−0.16, +0.81] MiB per image, 95% block-bootstrap intervals); growth above ~0.6–0.8 MiB per image would have been detected |
| scratch disk (`$FILAMENT_WORK`) | 6.9 GiB for prediction; about 11 GiB for the whole notebook (training smoke run and evaluation inputs included) |

## Reproduction tolerance

- **CPU stage, from the same network outputs:** byte-identical; the md5 above is reproduced.
- **GPU stages:** not deterministic from run to run, even on the same hardware (max |Δp| up to 1.3e-3 between two
  runs on the same inputs), so a from-images run is judged by agreement with the uploaded file, not by md5.
- **The rule** (`configs/FAST.json` `reproduction_tolerance`, `filament_seg.submission.compare`, applied by every
  `selector_stage apply`). A run passes when, compared with the uploaded file:
  - at most 3 test images differ in object count, each by at most 1 object;
  - the total object count is within ±3;
  - the pooled union-mask Dice over the images whose object count matches is ≥ 0.999;
  - and `verify_submission` passes.

  Reported as information, never deciding: whether the file is bit-identical (md5), whether every image has the same
  object count, the pooled Dice over all images, and the worst image Dice. **This rule was set on 2026-11-02, after the
  five measured runs below**, and replaces a stricter two-tier rule fixed before them, under which none of the four
  re-runs passed. It is therefore a description of the measured behaviour, not an independent test of it.
- **Measured behaviour.** Four independent GPU runs of the stored weights (same laptop GPU), each compared with the
  uploaded file, which is itself a fifth run. Each differs on 1–2 of the 180 images by one object; every other image
  agrees to Dice ≥ 0.9991. The cause is GPU non-determinism (|Δp| ≤ 1.3e-3) acting on a few borderline candidates whose
  selector score sits within about 1e-3 of the cut. In the uploaded file, a 2,932 px candidate on `20220523234912Lh`
  scores 0.2109 against the cut 0.2110 and is deleted; all four re-runs keep it. In the fresh clone, a 5,319 px
  candidate on `20220621185332Bh` also fell below the cut (0.2181 → 0.1916) after one pixel near p = 0.97 moved its
  minimum probability from 0.97006 to 0.97000. About 14 of the 1,745 test candidates change decision when their
  minimum probability moves by at most 3e-4.
- **Runs after the rule was set** (the rule was not changed): four more runs on the same laptop, 2026-11-06/07 (two
  executions of `pipeline.ipynb`, one `scripts/predict.py --from-images`, one during a runtime measurement), all within
  the rule: 1,186 objects, one image +1 (`20220523234912Lh`, the candidate above), pooled Dice over the count-matched
  images ≥ 0.999984. In all, the fresh clone and seven re-runs reproduce the uploaded file to one object on one or two
  images.
- **On a different system** (2026-11-07): on a rented RTX 4090 (Ada generation; driver 595.91.07) with only this README's
  instructions, `pipeline.ipynb` ran every cell without error (`TRAIN = False`). Its `FAST.csv` had 1,184 rows vs 1,185
  (one image, `20220621185332Bh`, had one object fewer); images with matching counts agreed at pooled Dice 0.99998, inside
  the rule. The evaluation gates the notebook runs passed and Table 1 came out as 0.673 / 0.591 / 0.444. Prediction took
  15.1 min (the networks 141 + 133 s on that GPU); the notebook took 23.5 min.

  | run | objects (uploaded 1,185) | images whose count differs | pooled Dice, count-matched images | pooled Dice, all images | worst count-matched image Dice | rule |
  |---|---|---|---|---|---|---|
  | GPU run, 2026-10-19 (a) | 1,186 | 1 (+1) | 0.999986 | 0.999368 | 0.99912 | pass |
  | GPU run, 2026-10-19 (b) | 1,186 | 1 (+1) | 0.999988 | 0.999370 | 0.99912 | pass |
  | GPU run, 2026-10-23 | 1,186 | 1 (+1) | 0.999986 | 0.999368 | 0.99912 | pass |
  | fresh clone, 2026-11-02 (reproduction run) | 1,185 | 2 (+1, −1) | 0.999986 | 0.998244 | 0.99977 | pass |

  None of them is bit-identical to the uploaded file, and none has the same count on every image. The fresh-clone file
  is in `tests/fixtures/reruns/`, classified by `tests/test_reproduction_tolerance.py`.

## Data and licences

- **Competition data:** "for non-commercial purposes only, including for participating in the Competition … and for
  academic research and education" (competition rules, §4a). It is **not redistributed** here: every stage reads it
  from your own download, the notebook included.
- **GONG H-alpha:** provided by NSO/AURA/NSF. The limb-darkening constants were measured on raw GONG frames early in
  the project (historical constants, kept fixed); only the constants are used.
- **Code:** MIT (`LICENSE`).
- **Model weights** (`weights/LICENSE-WEIGHTS.md`): the two nnU-Nets and the selector bag are our own models, trained
  from scratch, released under **MIT**. They were trained on the Competition Data, which the rules (§4a) allow to be
  used "for non-commercial purposes only, including for participating in the Competition … and for academic research
  and education". Users are responsible for their own use.

## How to cite

See `CITATION.cff`, and cite MAGFiLO (Ahmadzadeh et al., *Scientific Data* 11:1031, 2024) and nnU-Net
(Isensee et al., *Nature Methods* 18:203, 2021).

## Repository map

`directory_structure.txt`: the folder tree. `src/filament_seg/`: the package (`docs/module_map.md`). `configs/`: settings (`FAST.json`). `scripts/`: `prepare_data`,
`train`, `predict`, `eval_*`. `tests/`: unit tests and the CPU smoke test. `pipeline.ipynb`: the whole pipeline,
end to end. `docs/`: training recipe, run-root layout, module map. `weights/`: inventory and licences.
`eval_bundle/`: the out-of-fold evaluation inputs (derived numbers and our own masks).
`entry_points.md`, `REPRODUCE.md`.

Settings: `src/filament_seg/filament_paths.py` is the settings file; every path is an environment variable with a
repo-relative default. Outputs go only under `$FILAMENT_WORK` and `submissions/out/`. Assumption: the competition
download is unpacked at `$FILAMENT_KAGGLE_ROOT`. No stage reads MAGFiLO test annotations.
