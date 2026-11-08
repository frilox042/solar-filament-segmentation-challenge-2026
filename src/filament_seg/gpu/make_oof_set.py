#!/usr/bin/env python3
"""Assemble nnU-Net's per-fold validation predictions into one out-of-fold set.

Why this exists
---------------
Five trained folds are only a scoreable out-of-fold set if each fold's predictions lie entirely within its own
validation split and none lies in its training split. Globbing `fold_*/validation` and trusting the file names cannot
check that. This script derives ownership from `splits_final.json` instead and **asserts** the property, per fold,
before writing anything. A mismatch exits nonzero and names the fold; it does not repair it, because a fold whose
validation output is not its validation split means something is wrong upstream.

What it does
------------
1. Reads `splits_final.json`, nnU-Net's own 5-fold partition of the 707 training images.
2. For each fold, compares the PNG stems in `fold_<k>/validation/` against that fold's `val` list, **both
   directions**, and additionally reports any stem that falls in the fold's `train` list. A fold that is **still
   validating** is named as such rather than reported as a defect (see ``VALIDATION_DONE``).
3. Verifies the requested folds form a partition: no stem claimed twice.
4. Symlinks each prediction into `--out`, with absolute targets so the directory works from any cwd. Symlinks, not
   copies: 707 masks at 2048x2048 are not worth copying.

Refusals, both deliberate
-------------------------
* **A non-empty `--out` is refused outright**, with no `--force`, so an existing set is never silently grown or
  overwritten.
* **A partial set requires `--allow-partial`**: a subset averages to a plausible-looking number that is comparable to
  nothing.

Usage::

    python make_oof_set.py --out predictions/vessel_oof_full/by_image
    python make_oof_set.py --out /tmp/oof --folds 0,1,2,3 --allow-partial

The assembled set is then scored against the union labels in `data/competition_labels/<case>.png`
(`filament_seg.evaluation.records_png.union_mask`, uint8 0/1). In that mode ground-truth instances are **connected
components**, not annotation records, a coarser convention that must not be compared with per-record instance figures.
"""
import argparse
import json
import sys
from pathlib import Path

from filament_paths import NNUNET_PREPROCESSED, NNUNET_RESULTS, require

DEFAULT_DATASET = "Dataset007_FilamentVessel"
DEFAULT_TRAINER = "nnUNetTrainer_250epochs__nnUNetPlans__2d"

# nnU-Net writes this LAST, after every validation prediction, so its presence means the fold
# finished validating. Verified on folds 0-3 of the D007 ensemble: in each one `summary.json` is
# the newest file in `validation/`, stamped after the final PNG (fold 3: 00:13:28, the same
# second the trainer logged "Validation complete"). Treated as a hint, not a requirement -- a
# fold whose prediction set already matches its split is accepted without it, because the name
# belongs to nnU-Net and could change between versions.
VALIDATION_DONE = "summary.json"


def fold_ownership(splits, fold_ids=None):
    """Map stem -> owning fold, and fail loudly if two folds claim the same image.

    nnU-Net's splits are a partition by construction, but this is the assumption that makes
    "out-of-fold" mean anything, so it is checked rather than assumed. `fold_ids` names the fold
    each entry of `splits` corresponds to, so a clash is reported against the real fold number
    when only a subset of the file is being assembled.
    """
    owner = {}
    clashes = []
    ids = list(fold_ids) if fold_ids is not None else list(range(len(splits)))
    for k, f in zip(ids, splits):
        for stem in f["val"]:
            if stem in owner:
                clashes.append(f"{stem}: folds {owner[stem]} and {k}")
            owner[stem] = k
    return owner, clashes


def audit_fold(k, split, have):
    """Compare a fold's prediction stems against its validation split. Returns problems."""
    want = set(split["val"])
    train = set(split["train"])
    problems = []
    if missing := sorted(want - have):
        problems.append(f"fold {k}: {len(missing)} images in the validation split have no "
                        f"prediction: {missing[:4]}")
    # The §12d claim, checked as its own line so a failure names the right defect. A stem
    # from the fold's TRAIN split would mean the "predictions" are fitted output, not
    # out-of-fold, and every number scored from this set would be inflated.
    if leaked := sorted(have & train):
        problems.append(f"fold {k}: {len(leaked)} predictions are for images in this fold's "
                        f"TRAINING split, so the set is NOT out-of-fold: {leaked[:4]}")
    if stray := sorted(have - want - train):
        problems.append(f"fold {k}: {len(stray)} predictions are for images in neither split "
                        f"of this dataset: {stray[:4]}")
    return have & want, problems


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True,
                    help="directory to create; must not already hold files")
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--trainer", default=DEFAULT_TRAINER)
    ap.add_argument("--folds", default="",
                    help="comma-separated fold ids (default: every fold in splits_final.json)")
    ap.add_argument("--allow-partial", action="store_true",
                    help="assemble even though some folds are untrained. Any score from a "
                         "partial set is NOT comparable to a full-set result.")
    ap.add_argument("--from-dirs", default="",
                    help="assemble from arbitrary per-fold prediction directories instead of "
                         "nnU-Net's results tree, e.g. 'predictions/arch_A1/fold_%%d'. The "
                         "template must contain one %%d, which is filled with the fold id. Every "
                         "ownership check below still runs, unchanged -- this changes only where "
                         "the PNGs are read from, never whether they are audited.")
    ap.add_argument("--json", default="", help="write the audit report here")
    a = ap.parse_args()

    splits_path = require(NNUNET_PREPROCESSED / a.dataset / "splits_final.json",
                          f"nnU-Net splits for {a.dataset}",
                          "Run nnUNetv2_plan_and_preprocess for this dataset first.")
    splits = json.loads(splits_path.read_text())
    folds = ([int(x) for x in a.folds.split(",")] if a.folds else list(range(len(splits))))
    # Ownership is derived over the REQUESTED folds only. `splits_final.json` can hold entries
    # beyond the partition -- D007's (and D036's, copied from it) carries FOUR extra folds, 5-8,
    # each re-using fold 0's validation set with a different training set, appended for
    # seed/variant runs -- and computing ownership over the whole file registered those as 142
    # "clashes" against fold 0 when nothing had been requested from them (found 2026-09-18, on the
    # first five-fold assembly since they were appended). The docstring has always stated the
    # property as "the REQUESTED folds form a partition"; the code now checks what it says. With no
    # --folds every entry is requested, so the behaviour there is unchanged: a duplicated
    # validation image anywhere in the file is still refused.
    requested = [k for k in folds if k < len(splits)]
    owner, clashes = fold_ownership([splits[k] for k in requested], fold_ids=requested)
    not_requested = [k for k in range(len(splits)) if k not in requested]
    total = len(owner)

    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(
            f"--out {out} already holds {len(list(out.iterdir()))} entries and this script "
            f"will not touch it. An existing out-of-fold set backs recorded numbers; growing "
            f"it in place would invalidate them with no trace. Choose a new directory."
        )

    # Where a fold's predictions live. The default is nnU-Net's results tree; --from-dirs points
    # at predictions made outside it (an inference-setting arm, say). The AUDIT is identical in
    # both modes -- ownership still comes from splits_final.json and is still asserted per fold --
    # because the property being checked ("these predictions are that fold's held-out images") is
    # a claim about the split, not about which directory produced them.
    trainer_dir = NNUNET_RESULTS / a.dataset / a.trainer
    if a.from_dirs:
        if a.from_dirs.count("%d") != 1:
            raise SystemExit(f"--from-dirs must contain exactly one %d: {a.from_dirs!r}")

        def fold_val_dir(k):
            return Path(a.from_dirs % k)
    else:
        def fold_val_dir(k):
            return trainer_dir / f"fold_{k}" / "validation"

    # A missing trainer directory is NOT "every fold is untrained" -- it is almost always a
    # wrong --trainer, and reporting it as untrained sent a completed 8-fold run to the "no
    # predictions yet" branch (`--trainer nnUNetTrainer_250epochs`, without the
    # `__nnUNetPlans__2d` suffix this path needs). Name the real cause and list what is there.
    if not a.from_dirs and not trainer_dir.is_dir():
        sibs = sorted(p.name for p in (NNUNET_RESULTS / a.dataset).glob("*") if p.is_dir())
        raise SystemExit(
            f"no such trainer directory: {trainer_dir}\n"
            f"--trainer takes nnU-Net's full results directory name, including the plans and "
            f"configuration suffix (e.g. {DEFAULT_TRAINER}), not the bare trainer class.\n"
            + (f"available under {a.dataset}: {', '.join(sibs)}" if sibs else
               f"nothing is trained under {a.dataset} at all.")
        )
    problems = list(clashes)
    chosen, untrained, in_progress, unmarked = {}, [], [], []
    for k in folds:
        if k >= len(splits):
            problems.append(f"fold {k} requested but splits_final.json holds {len(splits)}")
            continue
        val_dir = fold_val_dir(k)
        if not val_dir.is_dir():
            untrained.append(k)
            continue
        have = {p.stem for p in sorted(val_dir.glob("*.png"))}
        # Distinguish "still validating" from "predictions are missing". Without this an
        # in-progress fold is reported as a DEFECT -- "141 images have no prediction" -- which
        # reads as a broken run when the truth is "wait 3 minutes". That is the state this
        # script will most often be run in, because the obvious moment to run it is as soon as
        # the last fold's training log stops moving.
        if not (val_dir / VALIDATION_DONE).exists():
            if have != set(splits[k]["val"]):
                in_progress.append(k)
                continue
            # Complete set, no marker: accept it rather than exclude a finished fold on the
            # strength of one filename, but say so -- the marker name is nnU-Net's, not ours.
            unmarked.append(k)
        stems, probs = audit_fold(k, splits[k], have)
        problems.extend(probs)
        for stem in sorted(stems):
            chosen[stem] = (val_dir / f"{stem}.png").resolve()

    partial = len(chosen) < total
    print(f"{a.dataset} / {a.trainer}")
    print(f"folds requested {folds}   untrained (no validation/ yet) {untrained or 'none'}")
    if not_requested:
        print(f"splits_final.json also holds entries {not_requested}, not requested: they take no "
              f"part in ownership and contribute no predictions")
    if in_progress:
        print(f"still validating (no {VALIDATION_DONE} yet, prediction set incomplete): "
              f"{in_progress} — not a defect, run again when they finish")
    if unmarked:
        print(f"complete but unmarked (no {VALIDATION_DONE}, prediction set matches the "
              f"split): {unmarked} — accepted")
    print(f"images covered {len(chosen)}/{total}")

    report = {
        "dataset": a.dataset, "trainer": a.trainer, "out": str(out),
        "folds_requested": folds, "splits_entries_not_requested": not_requested,
        "folds_untrained": untrained,
        "folds_in_progress": in_progress, "folds_complete_but_unmarked": unmarked,
        "images_covered": len(chosen), "images_total": total,
        "partial": partial, "problems": problems,
    }
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1))
        print(f"wrote {a.json}")

    if problems:
        print(f"\nFAILED ({len(problems)} problem{'s' if len(problems) != 1 else ''}):")
        for p in problems[:15]:
            print(f"  - {p}")
        if len(problems) > 15:
            print(f"  … and {len(problems) - 15} more")
        sys.exit(1)

    if partial and not a.allow_partial:
        raise SystemExit(
            f"\nonly {len(chosen)}/{total} images are available (untrained folds: "
            f"{untrained}). Refusing to assemble a partial set by default: a subset averages "
            f"to a plausible number that is comparable to nothing. Pass --allow-partial if "
            f"you want it as a correctness check, and stamp any score from it as partial."
        )

    out.mkdir(parents=True, exist_ok=True)
    for stem, target in sorted(chosen.items()):
        (out / f"{stem}.png").symlink_to(target)
    print(f"\nlinked {len(chosen)} predictions into {out}")
    if partial:
        print("⚠️  PARTIAL SET — any score from it is not comparable to a full-set result.")
    else:
        print("OK — every image in the dataset is covered, exactly once, by the fold that "
              "held it out.")


if __name__ == "__main__":
    main()
