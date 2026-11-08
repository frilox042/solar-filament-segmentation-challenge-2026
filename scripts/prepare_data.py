#!/usr/bin/env python3
"""Build inputs from the competition download only.

WIRED:
  --channels [--split train|test|both] [--datasets ...] [--limit N]
                      channel images of the four nnU-Net datasets from the raw competition JPEGs
                      ($FILAMENT_KAGGLE_ROOT/MAGFiLO_1.0_Kaggle_2026/...), resume-safe (filament_seg.prep.prepare)
  --check N           rebuild N images per split in memory and compare every channel with $FILAMENT_WORK's
  --selector-tables   WORK/analysis/selector_tables.npz: the 707 training case ids and the per-record COCO counts
                      (coco_k, coco_G) the FLOOR selector's fit needs. It is built from the competition training COCO
                      JSON ($FILAMENT_KAGGLE_COCO, default $FILAMENT_KAGGLE_ROOT/MAGFiLO_1.0_Kaggle_2026/train/...) and
                      the shipped case mapping (configs/case_mapping_full.json). It does not depend on --channels and
                      may run in any order. Needed only to REFIT the selector (`scripts/predict.py --refit`, the
                      full-retrain tier); the default predict path applies the released selector bags.

Training labels: `scripts/build_labels.py` (competition COCO -> labelsTr).
Splits and nnU-Net planning / training: `scripts/train.py` (filament_seg.training; docs/TRAINING.md).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selector-tables", action="store_true")
    a, rest = ap.parse_known_args()
    if not a.selector_tables:
        from filament_seg.prep import prepare
        return prepare.main(rest)
    from filament_seg import selector_tables as T
    return T.main([])


if __name__ == "__main__":
    sys.exit(main())
