# Run root ($FILAMENT_WORK) layout read by the CPU stage

Exactly the inputs listed in `tests/smoke/work_manifest.json`, at these relative paths.

```
WORK/
  analysis/selector_tables.npz                            GENERATED: scripts/prepare_data.py --selector-tables
  data/kaggle/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json
                                                          (competition training COCO, fit only: the default
                                                          $FILAMENT_KAGGLE_ROOT = WORK/data/kaggle; a real run sets
                                                          FILAMENT_KAGGLE_ROOT to the unpacked download instead)
  nnunet_env/nnUNet_raw/Dataset007_FilamentVessel/{imagesTr,imagesTs}/*_0000.png, case_mapping_full.json
  nnunet_env/nnUNet_raw/Dataset005_FilamentFull/case_mapping_full.json
  predictions/<OOF and test member dirs>                  (see docs/TRAINING.md §2)
  selector_scratch/<slot>/                                (written: bag pickle, PNGs, verify.json)
```

The `predictions/` subdirectory names are listed in `configs/<slot>.json` `cpu_stage`.

## Offline evaluation run root

`tests/smoke/stage_work.py --eval` stages the evaluation's inputs (`configs/eval/inputs.json`): the competition COCO,
the case mappings, `Dataset007/imagesTr`, the OOF base maps and the D036 member binaries, and
`predictions/vessel_oof_full/by_image` (the D007 >= 400 px reference arm, which Ours-Fast does not read). Outputs go to
`WORK/eval/` (override with `FILAMENT_EVAL`): `<slot>_oof_selection.pkl` (cached keeps) and one JSON per script.

**From the evaluation bundle** (no maps needed): `python -m filament_seg.evaluation.bundle stage --work WORK` writes the
same paths from `eval_bundle/` and your competition download: `analysis/oof_rows_FAST.npz` (the selector's OOF rows,
read by `evaluation/oof_selection.py` in place of the feature build), `predictions/vessel_oof_probs/` (uint16 maps
expanded from threshold levels, marked by `LEVEL_MAPS_FROM_EVAL_BUNDLE.json`; the selector fit refuses them),
`predictions/vessel_oof_full/by_image/`, `analysis/datastudy_diskfit.json`, the two case maps, and
`Dataset007_FilamentVessel/imagesTr/<case>_0000.png` (channel 0, from your training JPEGs).
