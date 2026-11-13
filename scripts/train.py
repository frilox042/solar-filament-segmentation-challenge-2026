#!/usr/bin/env python3
"""Training entry point (filament_seg.training): the full recipe of docs/TRAINING.md, smoke runs of it, or one step.

    python scripts/train.py full        [--root DIR] [--dry-run]   # ~65 GPU-h (10 fold trainings) + OOF maps + selector fit
    python scripts/train.py smoke       [--root DIR] [--dry-run]   # 10 cases, fold 0, 1 epoch: the notebook's quick check
    python scripts/train.py smoke-full  [--root DIR] [--dry-run]   # the full recipe on 10 cases, 5 folds x 1 epoch
    python scripts/train.py splits --dataset <Dataset> [--limit N]
    python scripts/train.py set-num-training --dataset <Dataset> --n N
Default run root: $FILAMENT_WORK/train. Models: <root>/nnunet_env/nnUNet_results, never the shipped weights.
Then: FILAMENT_WEIGHTS_NNUNET=<root>/nnunet_env/nnUNet_results python scripts/predict.py --from-images
      --bag-dir <root>/selector_scratch/nnHALO
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))


def main(argv=None) -> int:
    from filament_seg import training as T
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("full", "smoke", "smoke-full", "splits", "set-num-training"))
    ap.add_argument("--root", type=Path, help="training run root (default $FILAMENT_WORK/train)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dataset")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n", type=int)
    a = ap.parse_args(argv)
    if a.cmd == "splits":
        print("wrote", T.write_splits(a.dataset, a.limit))
        return 0
    if a.cmd == "set-num-training":
        print("wrote", T.set_num_training(a.dataset, a.n))
        return 0
    steps = {"full": T.full_recipe, "smoke": T.smoke_recipe, "smoke-full": T.smoke_full_recipe}[a.cmd](a.root)
    return T.run(steps, a.root, dry_run=a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
