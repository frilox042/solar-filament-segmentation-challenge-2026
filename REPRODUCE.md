# REPRODUCE: every reported number, and the command or file that produces it

Status column:
- **command**: reproduced by a command in this repository (output named).
- **code**: a parameter that is literally in this repository's code or config (location named).
- **bundle**: reproduced from the out-of-fold evaluation bundle tracked in `eval_bundle/` (14 MB) plus your own
  competition download, CPU only, no retraining (Evaluation, below). The bundle is derived from the models'
  out-of-fold maps; regenerating those needs the full retrain (`docs/TRAINING.md`), and the same commands then run on
  the full maps.
- **GPU**: a runtime or memory measurement.

Commands assume `FILAMENT_WORK` was populated with `scripts/prepare_data.py`, `tests/smoke/stage_work.py`, or (for the
evaluation) `python -m filament_seg.evaluation.bundle stage`
(layout: `docs/work_layout.md`).

## Submission files

| number | status | command → output |
|---|---|---|
| FAST file md5 `7c5bd00d…`, 1185 rows | command | `python tests/smoke/run_cpu_smoke.py --python … --source-root … --out OUT` → `OUT/smoke_result.json` (also `scripts/predict.py --slot FAST` → `submissions/out/FAST.csv`). Byte-identical from the network outputs |
| FAST from raw images, fresh clone, same laptop GPU, 2026-11-02 (reproduction run; the runtime reference is now the 2026-11-07 run below): 1,185 / 1,185 objects, count differs on 2 images (+1, −1), pooled Dice 0.999986 over the 178 count-matched images (0.998244 over all 180) → rule **passes**; not bit-identical | GPU | `scripts/predict.py --slot FAST --from-images` (31.8 min end to end with the pipeline of that date, which also built the unused Dataset005 channels, one image at a time). Output kept as `tests/fixtures/reruns/FAST_fresh_clone.csv`; `python -m filament_seg.submission.compare tests/fixtures/reruns/FAST_fresh_clone.csv submissions/FAST.csv --slot FAST` |
| different system: rented RTX 4090 (Ada), driver 595.91.07, README only, `pipeline.ipynb` (TRAIN = False): 1,184 vs 1,185 rows, one image −1, pooled Dice 0.99998 count-matched → rule passes; Table 1 reproduced; prediction 15.1 min | GPU | README, "Reproduction tolerance" |
| four runs after the rule was set (2026-11-06/07: two notebook executions, one `predict.py --from-images`, one runtime run): 1,186 objects each, one image +1, pooled Dice ≥ 0.999984 count-matched → rule passes | GPU | README, "Reproduction tolerance" |
| three further GPU runs of the same two networks, released selector applied: 1,186 objects each, one image +1, pooled Dice 0.999986–0.999988 count-matched (0.999368–0.999370 all images) → rule passes | GPU | README, "Reproduction tolerance" (table) |

The rule is in the README, "Reproduction tolerance". It was set on 2026-11-02, after these five measured runs (the
uploaded file and the four re-runs above), so it describes their measured behaviour rather than testing it.

## Data and split

| number | status | command → output |
|---|---|---|
| 707 training observations | command | `python scripts/report_data_numbers.py` → `training_observations` |
| one to three annotators; 296 with two or three records | command | same → `records_per_observation` {1: 411, 2: 145, 3: 151}, `observations_with_2_or_3_records` = 296 |
| 2048 × 2048 px, 8-bit grey | code | `filament_seg.maps.load_u16` asserts (2048, 2048); `prep.prepare.raw_u8` reads JPEG → mode L |
| folds: every fifth sorted case, 5 folds | code | `docs/TRAINING.md` §0 (`cases[i::5]`) |
| training labels: union / barb class from the competition COCO, equal to the labels of the shipped members on 707 / 707 | command | `python scripts/build_labels.py`; check `python tests/eval_records/build_labels_check.py --reference-raw <nnUNet_raw of the shipped members> --out …` → `tests/eval_records/build_labels_record.json` |

## Methodology

| number | status | where |
|---|---|---|
| CLAHE kernel 256, clip 0.01; Gaussian σ 64; coherence σ 3; Sato σ 16, disk c0 > 0.08·max | code, command | `filament_seg/prep/channels.py` (`four_channels`, `sato_channel`); array-equal to the stored channels: `python scripts/prepare_data.py --check 3 --reference …` |
| limb law 0.077, 0.358; radius 903.2 px | code | `prep/channels.py` `A1, A2, R_LIMB` |
| min–max to 16 bit | code | `prep/channels.py` `to_u16` |
| two networks, fold 0: nnU-Net on c0–c4, nnU-Net on c0–c5 (3 classes) | code | `configs/FAST.json`; `configs/datasets/*.dataset.json` |
| stock trainer, 250 epochs, stock 2D plans | code | `docs/TRAINING.md` §1; `weights/README.md` |
| out-of-fold maps: tile step 0.75, mirror TTA | code | `docs/TRAINING.md` §2 (`predict_oof_probs.py --step-size 0.75`) |
| candidates: p ≥ 0.97 | code | `filament_seg/candidates.py` `base_mask("t97")`; `configs/FAST.json` `cpu_stage.core_t` |
| 4-connected objects ≥ 100 px | code | `candidates.py` `FLOOR_MIN`, `BIG_MIN` |
| 22 features = 9 + 8 + 5 | code | `selector_features.py` `BASE_FEATS`, `CTX_FEATS`, `step2_matrix` (k members → 2k + 4 member columns: 2 → 8) |
| target = Σ hit IoU (> 0.5) / k records | code, command | `scorer.py` `coco_targets`, `Ref.u`; `tests/test_scorer.py` (host rule) |
| bag of 5 HGB: 250 iterations, 12 leaves, lr 0.05, ≥ 20 per leaf; each on 4/5 of the observations | code | `selector_fit.py` `HGB`, `fit_bag` |
| cut 0.211 = PQ(w 0.5) of candidates ≥ 400 px / 2 | command | `python -m filament_seg.selector_stage fit` prints `two_cut` 0.42199336931468706 (cut 0.21100) |
| isotonic confidence, cross-fitted, not on the selection path | bundle | `scripts/eval_calibration.py` (`filament_seg/evaluation/calibration.py`); it is never read by `selector_stage` |
| test inference: fold 0, tile step 0.5, mirroring; 2 member masks (6-ch nnU-Net fold 0; P = the base map at p ≥ 0.5, no network) | code | `configs/FAST.json` `cpu_stage.members`; `gpu/pipeline.py` `plan("FAST")`; `selector_features.member_masks` |
| instances ≥ 20 px, 4-connected, COCO RLE, verified | code, command | `configs/FAST.json` `min_area_csv`; `submission/rle.py`; `tests/test_rle.py`; `submission/verify.py` |
| 180 test observations | command | `configs/case_mapping_full.json` ["test"]; `render.test_setup` asserts 180 |

## Evaluation

Each command checks its own output against the reported values in `configs/eval/expected.json` and exits non-zero on
any mismatch; each has a negative control. Run them in this order, CPU only (`OMP_NUM_THREADS=4`, peak < 4 GiB):

```bash
export FILAMENT_KAGGLE_ROOT=/path/to/unzipped/filament-segmentation-2026 FILAMENT_WORK=$PWD/work_eval
python -m filament_seg.evaluation.bundle stage --work $FILAMENT_WORK       # from eval_bundle/ + your download, ~5.5 min,
                                                                           # needs ~2.7 GB free in WORK (channel 0 is 2.5 GB)
# (or, with the full out-of-fold maps: python tests/smoke/stage_work.py --source-root SRC --work $FILAMENT_WORK --eval)
python scripts/eval_oof_selection.py               # cross-fitted keeps -> WORK/eval/FAST_oof_selection.pkl, ~4 min
python scripts/eval_heldout_annotator.py            # -> WORK/eval/heldout_annotator.json, ~2 min
python scripts/eval_morphometry.py                  # -> WORK/eval/morphometry.json, ~2.5 min
python scripts/eval_calibration.py                  # -> WORK/eval/calibration_FAST.json, ~1 min
python scripts/eval_annotator_agreement.py          # -> WORK/eval/annotator_agreement.json, ~2 min (no model)
python scripts/eval_never_seen.py                   # -> WORK/eval/never_seen.json + never_seen_rows.npz, ~7 min
```

**The evaluation bundle** (`eval_bundle/`, `filament_seg/evaluation/bundle.py`; every file in
`eval_bundle/MANIFEST.sha256`, checked before staging). It holds only model-side, derived inputs:

| file | size | what | read by |
|---|---|---|---|
| `oof_rows_FAST.npz` | 0.7 MB | the selector's 6,747 out-of-fold rows: 22 features per candidate, area, and per-record match numbers (IoU sums, hit counts, records hit / missed; per-image record and annotation counts). No geometry | `oof_selection` in place of the feature build |
| `oof_levels/<case>.png` (707) | 8.9 MB | the base network's OOF probability map reduced to the five thresholds the evaluation applies (p ≥ 0.97 candidates, ≥ 0.92 negative control, < 0.3 barbs, < 0.1 and > 0.5 never-seen classes) | candidates, morphometry, never-seen |
| `d007_oof_bin/<case>.png` (707) | 4.0 MB | the D007 OOF argmax binaries (our predictions), unchanged | the D007 ≥ 400 px reference arm |
| `diskfit.json` | 0.1 MB | per-frame limb fit (centre, radius; image-derived) | never-seen r/R |

Images and annotations are not in it (competition data, not redistributed): staging reads the training COCO and the 707
training JPEGs from your download and writes channel 0 (the min-max uint16 of the raw frame, as `prep.channels`) for the
never-seen contrast and clutter features. The level maps are expanded to uint16 maps in which every threshold above
gives the same answer as on the real map (`tests/test_eval_bundle.py`, with a negative control at a threshold the
bundle does not keep), so the scripts run unchanged; the selector fit refuses them. Two differences from a run on the
full maps: the `p_max` column of `never_seen_rows.npz` holds level values, not probabilities, and `--records png` /
`--reference-rows` (historical replications) need inputs the bundle does not hold. Every gate passes from the bundle,
and every output equals the run on the full maps leaf for leaf (all five result JSONs, the cached keeps and scores), except
that `p_max` column (verified 2026-11-02; staging plus the six scripts about 19 min on the reference laptop CPU, times
above). `python -m filament_seg.evaluation.bundle export --out DIR` rebuilds the bundle from the full maps.

With the full maps instead, SRC is a directory laid out like WORK that holds the out-of-fold maps. `eval_calibration.py --test-bag PKL` also reports
(not gated) the mean score of the kept test objects under a bag, which needs the test network outputs in WORK. The keeps cache stores the identity of its inputs and refuses a stale cache; `eval_oof_selection.py
--rebuild` recomputes it. If the members are retrained, re-stage the new out-of-fold maps, write a new
`expected.json` block, and regenerate the reported numbers from the outputs.

**Annotator records.** The last two scripts read each annotator record's mask, rasterised by default from the
competition training COCO (`--records coco`, `filament_seg/evaluation/records_coco.py`), the only label file the
competition ships. The union of the records of an observation equals the shipped training label on 707 / 707 images,
bit for bit (`tests/test_records_coco.py`, with a negative control that drops one record).

The report quotes the values of the default `--records coco` run (never-seen filaments: 523 for the submission, contrast
AUC 0.8990, support AUC 0.8549; for the single network 522, 0.884 and 0.861). An earlier measurement used a PNG export of the same records (`extracted/training/<record>/annotation/*.png`), half a pixel off the COCO rasterisation and
not part of the competition download; `--records png` replicates the earlier single-network values (and gives 516, 0.8999,
0.8499 for the submission) and needs that export. The two sources differ by at most 0.009 on any AUC or rate (most by < 0.003), and counts move by a few
objects (existence 0.8398 vs 0.8393, 8 of 287 vs 8 of 284).

Keys refer to the output JSON. `S` = `results.seed_20261018`, the bootstrap seed of the reported intervals.

| number | status | command → output key |
|---|---|---|
| 743 held-out-record units; 296 observations | bundle | `eval_heldout_annotator.py` → `S.n_units`, `S.n_images`; also `scripts/report_data_numbers.py` |
| Table 1 PQ 0.340 / 0.444 / 0.409 (annotator / submission / single network), Δ +0.104 [+0.093, +0.115], +0.069 [+0.058, +0.081] | bundle | `eval_heldout_annotator.py` → `S.pooled.<FAST\|D007>_vs_H_A0.pq_w05` (`reference` = annotator) |
| Table 1 Dice 0.616 / 0.673 / 0.674, Δ +0.057 [+0.044, +0.069], +0.058 [+0.048, +0.068] | bundle | same → `S.dice` (`*_image_mean`, `*_minus_H`) |
| Table 1 IoU 0.517 / 0.591 / 0.566 and Δ | bundle | same → `S.pooled.*_vs_H_A0.mean_iou` |
| Table 1 Pred. 0:1 0.178 / 0.155 / 0.236 and Δ; GT 1:≥2 0.032 / 0.035 / 0.053 and Δ | bundle | same → `S.pooled.*_vs_H_A0.pred_0to1`, `.gt_1toge2` |
| splits more vs an annotator restricted to ≥ 400 px, +0.009 [+0.002, +0.016] | bundle | same → `S.pooled.FAST_vs_H_A400.gt_1toge2` |
| PQ 0.444 vs 0.340; ×1.31 | bundle | same; ×1.31 = the ratio of the two points |
| 49 observations ≥ 4 days from every training observation, PQ +0.110 | bundle | same → `S.far` |
| ECE 0.0138 vs raw 0.285 | bundle | `eval_calibration.py` → `results.ece_isotonic_oof_bagmean`, `results.ece_raw_bagmean` |
| width +6.0% (0.67 px of 11.07 px), single network +13.8%; absolute error 2.39 px | bundle | `eval_morphometry.py` → `results.width` |
| branching −58.9% [−62.0, −55.8] (single network −56.0%) | bundle | same → `results.branching` |
| barb pixels covered 60.0% vs 69.8% (single network ≥ 400 px) | bundle | same → `results.barbs` (`barb_coverage_per_image_mean`) |
| split share 0.0496 → 0.0321; coverage 0.64 → 0.45 | bundle | same → `results.split`, `results.selector_deletion_bag0.FAST` |
| existence 0.8398 vs IoU-0.5 match 0.5656 (intervals [0.8249, 0.8537] / [0.5390, 0.5926]) | command | `eval_annotator_agreement.py` → `results.existence`, `.matching`, `.existence_minus_matching`. Also 4,290 matched pairs, median IoU 0.6292, pooled +0.2866. `--records png` replicates the earlier 0.8393 / 0.5629 |
| 523 never-seen filaments (p < 0.1); contrast AUC 0.8990; support AUC 0.8549 | bundle | `eval_never_seen.py` → `results.per_model.FAST` (`n_D_bag_mean`, `contrast_AUC_bag_mean`, `support_AUC_bag_mean`; every bag too). `--records png` gives 516 / 0.8999 / 0.8499 |
| single network (D007): 522 never-seen filaments, contrast AUC 0.884 [0.864, 0.905], support AUC 0.861 [0.833, 0.888] | bundle | same → `results.results.D.contrast`, `.support`. Interior texture (0.688), clutter, width and r/R are computed by the same script |
| 8 of 287 drawn by every annotator vs 78.5% | bundle | same → `results.headline` (`D_all_agree`, `D_multi`, `M_all_agree_frac`; k = 2 / 3 splits) |
| … vs 50.2% at equal area | not reproduced | this code's pre-declared area binning gives 0.5026 (50.3%); `not_reproduced` in `configs/eval/expected.json` |

## Cost

Reference run: `scripts/predict.py --slot FAST --from-images`, 180 test observations, reference hardware
(README, "Runtime and memory"); stage times from the output files' modification times.

| number | status | source |
|---|---|---|
| end to end 19.0 min (1,141 s); GPU throttled for most of each network step (README) | GPU | reference run (2026-11-07), `scripts/predict.py --from-images` prints `{"step", "seconds"}` per step |
| CPU: input channels 3.9 min (234 s; Dataset007 + Dataset036, 4 workers) | GPU | same run |
| GPU: Dataset007 fold 0 6.3 min (378 s); Dataset036 fold 0 6.7 min (402 s); 1.2 s per case at full clock | GPU | same run |
| CPU: selector (with member P), CSV, verify, tolerance check 2.1 min (127 s) | command | same run; `python tests/smoke/run_cpu_smoke.py` prints per-step seconds |
| host peak 5.8 GiB (summed PSS; RSS 7.9 GiB); GPU 1.7 GiB per process; no memory growth (PSS slopes in README); scratch disk 6.9 GiB | GPU | same run, sampled every ~6 s |
