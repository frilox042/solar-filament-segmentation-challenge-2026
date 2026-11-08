#!/usr/bin/env python3
"""Build the nnU-Net members' training labels from the competition download (full-retrain tier).

    python scripts/build_labels.py [--datasets D005 D007 D012 D036] [--limit N] [--workers 4]
    python scripts/build_labels.py --verify-against <nnUNet_raw tree>    # compare, write nothing

Reads only the competition training COCO JSON ($FILAMENT_KAGGLE_COCO) and the train_images listing
($FILAMENT_KAGGLE_ROOT/MAGFiLO_1.0_Kaggle_2026/train/train_images). Writes
$nnUNet_raw/<Dataset>/labelsTr/img_XXXX.png: union labels (0/1) for Dataset005/007/012 and the barb-class label
(0/1/2) for Dataset036. Channels and dataset.json come from `scripts/prepare_data.py --channels`.
Method and identity record: `filament_seg.prep.labels`. Memory: about 1 GiB per worker.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from filament_seg.prep import labels as L  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(L.DATASETS), choices=list(L.DATASETS))
    ap.add_argument("--limit", type=int, default=0, help="first N cases only (0: all 707)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--kaggle-root", type=Path)
    ap.add_argument("--coco", type=Path)
    ap.add_argument("--raw", type=Path, help="nnUNet_raw tree to write (default: $nnUNet_raw)")
    ap.add_argument("--verify-against", type=Path, help="compare with this nnUNet_raw tree instead of writing")
    a = ap.parse_args(argv)
    if a.verify_against:
        res = L.verify(a.datasets, a.verify_against, a.coco, a.kaggle_root, a.limit, a.workers)
        summ = {L.DATASETS[k]: {"n_cases": len(res), "n_equal": sum(r[k]["equal"] for r in res.values()),
                                "max_diff_px": max(r[k]["n_diff"] for r in res.values())} for k in a.datasets}
        print(json.dumps(summ, indent=1))
        return 0 if all(s["n_equal"] == s["n_cases"] for s in summ.values()) else 1
    print(json.dumps(L.build(a.datasets, a.coco, a.kaggle_root, a.raw, a.limit, a.workers), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
