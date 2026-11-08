#!/usr/bin/env python3
"""Predict entry point for nnHALO (slot FAST), the submitted model.

  python scripts/predict.py --slot FAST [--bag-dir DIR | --refit]                                    # CPU stage only
  python scripts/predict.py --slot FAST --from-images [--limit N] [--device cuda] [--dry-run]
                            [--bag-dir DIR | --refit]

`--from-images` runs the whole chain from the raw competition test JPEGs: channels, then GPU member predictions,
then the CPU stage. Each step is resume-safe. See `filament_seg.gpu.pipeline` for the interpreters and weights it
expects. `--dry-run` prints the plan and runs nothing. `--limit N` restricts the channel and GPU steps to the first
N test images and skips the CPU stage, which needs all 180.

The CPU stage (base mask -> FLOOR selector -> CSV, `filament_seg.selector_stage`):
- default: APPLY the released selector bag, `weights/selector/FAST_bag.pkl` (our own trained model, listed with
  its sha256 in `weights/MANIFEST.sha256`; checked before loading). Needs only the network outputs.
- `--bag-dir DIR`: apply DIR/FAST_bag.pkl instead.
- `--refit` (full-retrain tier): re-fit the selector on the 707 out-of-fold rows, then apply it. Needs the
  out-of-fold bundle in $FILAMENT_WORK (docs/work_layout.md) and the competition COCO (selector tables).
Without `--from-images`, $FILAMENT_WORK must already hold the network outputs (docs/work_layout.md).

Each run compares its CSV with the uploaded file under the reproduction rule (`configs/FAST.json`
`reproduction_tolerance`, README "Reproduction tolerance": at most 3 images differ by one object, total within +-3,
pooled Dice over count-matched images >= 0.999), reports whether the file is bit-identical and whether every
image has the same object count, and exits non-zero only when the step fails. The md5 match is reported separately,
not required.

The CPU stage runs in its own process (the submission step resets the BLAS thread variables on import).

Output goes to submissions/out/FAST.csv. The uploaded file submissions/FAST.csv is never overwritten.
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "src"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slot", default="FAST", choices=("FAST",), help="nnHALO, the submitted model (default)")
    sel = ap.add_mutually_exclusive_group()
    sel.add_argument("--bag-dir", help="apply DIR/<slot>_bag.pkl instead of the released selector bag")
    sel.add_argument("--refit", action="store_true",
                     help="full-retrain tier: re-fit the selector on the 707 out-of-fold rows, then apply it")
    ap.add_argument("--from-images", action="store_true", help="channels -> GPU networks -> CPU stage")
    ap.add_argument("--limit", type=int, default=0, help="--from-images: first N test images only (no CPU stage)")
    ap.add_argument("--device", default="cuda", help="--from-images: torch device for the members")
    ap.add_argument("--dry-run", action="store_true", help="--from-images: print the plan, run nothing")
    a = ap.parse_args(argv)
    sys.path.insert(0, SRC)
    from filament_seg import filament_paths as fp
    if a.from_images:
        from filament_seg import weights
        from filament_seg.gpu import pipeline
        if not a.dry_run:
            weights.assemble()          # the weights stored in the repository (weights/fast/), verified against the manifest
        return pipeline.run([a.slot], a.limit, a.device, a.dry_run, refit=a.refit, bag_dir=a.bag_dir)
    slot = a.slot
    args = ["both" if a.refit else "apply", "--slot", slot, "--csv", str(fp.SUBMISSIONS_OUT / f"{slot}.csv")]
    if a.bag_dir:
        args += ["--bag", os.path.join(os.path.abspath(a.bag_dir), f"{slot}_bag.pkl")]
    print(f"== {slot}: {json.dumps(args)}", flush=True)
    code = (f"import sys; sys.path.insert(0, {SRC!r}); "
            "from filament_seg.selector_stage import main; sys.exit(main())")
    return subprocess.run([sys.executable, "-c", code, *args]).returncode


if __name__ == "__main__":
    sys.exit(main())
