# Entry points

| command | does |
|---|---|
| `python scripts/prepare_data.py --channels [--split train\|test\|both] [--limit N]` | network input channels from the raw competition JPEGs |
| `python scripts/prepare_data.py --check N --reference DIR` | rebuild N images per split in memory and compare |
| `python scripts/prepare_data.py --selector-tables` | selector tables from the competition COCO |
| `python scripts/predict.py [--slot nnHALO] [--bag-dir DIR \| --refit]` | nnHALO's CPU stage from the network outputs in `$FILAMENT_WORK` |
| `python scripts/predict.py [--slot nnHALO] --from-images [--limit N] [--dry-run] [--device cuda]` | nnHALO: channels -> GPU networks -> CPU stage, resume-safe |
| `python -m filament_seg.selector_stage fit\|apply\|both` | the CPU stage directly |
| `python scripts/train.py full \| smoke \| smoke-full \| splits` | the training recipe of `docs/TRAINING.md` as resumable steps (full: ~65 GPU-h; smoke: 10 cases, 1 epoch; smoke-full: the full recipe on 10 cases) |
| `python tests/smoke/run_cpu_smoke.py ...` | byte-exact reproduction of the uploaded nnHALO file, plus negative controls |
| `python scripts/report_data_numbers.py` | data counts quoted in the report (707 / 296 / 743) from the competition COCO |
| `python examples/build_notebook.py` | regenerate and execute `pipeline.ipynb`: the whole pipeline (weights, optional training + smoke run, the 180 test observations, one observation in detail, the evaluation, runtime); needs `FILAMENT_KAGGLE_ROOT` and the two environments |
| `python scripts/assemble_weights.py` | join the stored weight parts (`weights/nnHALO/`) into `$FILAMENT_WEIGHTS` and check their hashes |
| `python -m filament_seg.evaluation.bundle stage --work W` | stage the offline evaluation's inputs from the tracked `eval_bundle/` and your competition download (no maps needed) |
| `python tests/smoke/stage_work.py --source-root SRC --work W --eval` | stage them from the full out-of-fold maps instead (`configs/eval/inputs.json`) |
| `python scripts/eval_oof_selection.py [--reference-keeps nnHALO=PKL]` | cross-fitted selector keeps on the 707 OOF rows (input of every evaluation number) |
| `python scripts/eval_heldout_annotator.py` | Table 1: held-out annotator vs nnHALO and D007 on the host metric |
| `python scripts/eval_morphometry.py` | Morphology: width, branching, split degree, barb coverage of the kept masks |
| `python scripts/eval_calibration.py [--test-bag PKL]` | Calibration: ECE of the selector score, test score scale |
