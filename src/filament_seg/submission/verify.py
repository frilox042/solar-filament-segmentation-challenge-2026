#!/usr/bin/env python3
"""Pre-submission gate: decode a submission CSV back and prove it says what we meant.

Why this exists
---------------
This project shipped **eight** Kaggle submissions before anything ever checked the
submission encoder end-to-end, and the reason is worth stating plainly: our offline
yardstick (``honest_test_eval.py``) scores the *prediction PNGs* against ground truth. It
never touches ``make_submission.py``. So the entire RLE path — instance separation, COCO
encoding, the CSV writer — was unvalidated, and a fault there would be **invisible offline
and catastrophic on the leaderboard**.

Two such faults were live candidates, and both are the kind that pass every superficial
check:

* **A wrong RLE size.** The spec fixes 2048x2048. Another team's public notebook encodes
  from a 512x512 prediction, which is where the ``PPP8`` string in the community's
  empty-mask reports comes from. Measured cost of that mistake on
  our own predictions: Dice **0.0067** instead of 0.6900.
* **A transposed mask.** COCO RLE is column-major, so a missing ``asfortranarray`` swaps the
  axes. Because the images are **square**, every shape and size assertion still passes.
  Calibrated here: ``Dice(mask, mask.T)`` on a real prediction is **0.0000**.

Neither was present — measured mean Dice between the decoded submission and the intended
masks is **0.999712**, with **zero** invented pixels and every difference a connected
component below ``--min-area``. This script is what establishes that, and it is cheap enough
to run before every submission.

What it checks
--------------
1. **Every RLE decodes** at 2048x2048. A CR-suffixed RLE raises ``ValueError`` here, which is
   the failure mode ``tests/test_submission_format.py`` guards against at the writer.
2. **LF line endings.** ``csv.writer``'s default dialect emits CRLF; see §12a-1.
3. **No invented foreground.** Decoded pixels must be a subset of the source prediction. A
   nonzero count means the encoder is fabricating structure.
4. **Every dropped pixel is explained** by the ``--min-area`` instance filter, so a
   geometric bug cannot hide behind an intentional filter.
5. **Row/id sanity** — ids parse as ``<image_id>_<n>``, image ids match the prediction
   filenames, and no id repeats.
6. **No image is silently absent.** Every prediction PNG holding foreground must appear in
   the CSV. This check was **missing from the first version of this script**, which iterated
   the CSV and therefore could only ever inspect images the CSV mentioned — a blind spot in a
   tool whose whole purpose is catching silent loss. An image absent from the submission is
   scored as a prediction of nothing, so dropping one throws away pure recall. Images that
   are absent *and* genuinely empty are reported as fine, not as failures (there is exactly
   one in the current test set). An image whose foreground is ONLY components below
   ``--min-area`` is emptied by the instance filter (rule 4), so it is reported
   (``absent_from_csv_below_min_area``) and not failed; one component >= ``--min-area`` fails.

Usage::

    python verify_submission.py submission_vessel.csv predictions/vessel_full/by_image
    python verify_submission.py sub.csv preds/ --min-area 20 --json out.json

Exit status is nonzero if any check fails, so it can gate a submission in a shell script.

⚠️ **Run this in the nnU-Net virtualenv**, which is where ``pycocotools``, ``scipy`` and
``PIL`` live. ``pycocotools`` is imported **at point of use** inside :func:`decode_rows`, not
at module scope, so the module imports — and everything that does not actually decode an RLE
runs — under the interpreter the test suite uses, which lacks it. That split between
environments is the subject of ``tests/test_gt_free_imports.py``.

**What is covered, and what is not.** ``tests/test_verify_submission.py`` drives :func:`check`
on hand-built masks with a negative control for every failure it can report, and drives the
CLI on a CRLF file and on a malformed-row file. **The decode call itself remains
hand-checked** — it is one delegated ``pycocotools`` call, and no interpreter here has both
``pytest`` and ``pycocotools`` — exercised against a CRLF submission (fails on line endings
alone) and its LF regeneration (passes).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

SIZE = 2048          # the competition spec fixes this for every image
Image.MAX_IMAGE_PIXELS = None


def decode_rows(csv_path):
    """Group decoded instance masks by image id. Returns (per_image, problems, ids)."""
    overlaps: list = []
    per_image, problems, ids = {}, [], []
    # a fragmented mask's COCO RLE can exceed csv's 128 KiB default field limit; the gate must decode it, not crash
    # (otherwise a crash here would refuse a valid file for the wrong reason)
    csv.field_size_limit(1 << 30)
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header != ["filament_id", "segmentation_rle"]:
            problems.append(f"unexpected header: {header!r}")
        for lineno, row in enumerate(reader, start=2):
            if len(row) != 2:
                problems.append(f"line {lineno}: expected 2 fields, got {len(row)}")
                continue
            fid, rle = row
            ids.append(fid)
            if "_" not in fid:
                problems.append(f"line {lineno}: id {fid!r} has no _<n> suffix")
                continue
            image_id = fid.rsplit("_", 1)[0]
            # Imported here, at point of use, and deliberately AFTER the two checks above:
            # they `continue` before any decode, so a CSV of malformed rows can be verified
            # by an interpreter with no pycocotools -- which is the one the test suite runs
            # under. The import is cached, so the per-row cost is a sys.modules lookup. It
            # sits outside the try so a missing pycocotools raises plainly instead of being
            # reported 2,000 times as "this row failed to decode".
            from pycocotools import mask as coco_mask
            try:
                d = coco_mask.decode({"size": [SIZE, SIZE],
                                      "counts": rle.encode("ascii", "strict")}).astype(bool)
            except Exception as e:                      # noqa: BLE001 - reported, not raised
                problems.append(f"line {lineno}: {fid} RLE failed to decode: "
                                f"{type(e).__name__}: {e}")
                continue
            acc = per_image.get(image_id)
            if acc is None:
                per_image[image_id] = d
            else:
                # ⛔ ROWS MUST BE DISJOINT, AND NOTHING HERE CHECKED THAT UNTIL 2026-08-15.
                # Every downstream check compares the UNION of an image's rows against the
                # prediction PNG, and a union is INVARIANT to how the rows partition it. An
                # adversarial sweep demonstrated the consequence: a 5-row CSV whose rows are five
                # byte-identical copies of the whole mask passed this gate at pixel Dice 1.000000,
                # 0 invented pixels, 0 dropped components, exit 0 — i.e. the pre-submission gate
                # waved through the duplication shape the competition rules rule out. Overlap is
                # counted and reported as a failure.
                overlap = int(np.logical_and(acc, d).sum())
                if overlap:
                    overlaps.append((image_id, fid, overlap))
                acc |= d
    return per_image, problems, ids, overlaps


def check(per_image, ids, problems, n_cr, pred_dir, min_area, csv_path="", overlaps=None):
    """Compare decoded masks against the predictions they were built from.

    Split out of :func:`main` so it is reachable from a test: every check below runs on
    plain numpy masks and needs no ``pycocotools``, which the interpreter running the suite
    does not have. ``tests/test_verify_submission.py`` drives it directly.

    Returns ``(report, fail, absent_ok)`` — the JSON report, the list of failure strings
    (empty means the submission passed), and the ids absent from the CSV whose prediction is
    genuinely empty, which is reported to the reader as fine rather than as a failure.
    """
    fail = []
    if n_cr:
        fail.append(f"{n_cr} CR bytes: written with CRLF. Every RLE then ends in \\r, which "
                    f"pycocotools refuses to decode (it is a known failure of CRLF files)")

    fail.extend(problems)
    dupes = len(ids) - len(set(ids))
    if dupes:
        fail.append(f"{dupes} duplicate filament_id values")

    invented = 0
    dropped_sizes = []
    dices = []
    missing = []
    for image_id, dec in sorted(per_image.items()):
        p = pred_dir / f"{image_id}.png"
        if not p.is_file():
            missing.append(image_id)
            continue
        orig = np.asarray(Image.open(p)) > 0
        if orig.shape != dec.shape:
            fail.append(f"{image_id}: prediction is {orig.shape}, spec requires "
                        f"{(SIZE, SIZE)} — the RLE size is wrong")
            continue
        invented += int((dec & ~orig).sum())
        only_orig = orig & ~dec
        if only_orig.any():
            # scipy's stubs model the 2-tuple return as an int, hence the ignore.
            lab = ndimage.label(only_orig)[0]  # type: ignore[index]
            dropped_sizes.extend(np.bincount(np.asarray(lab).ravel())[1:].tolist())
        s = int(orig.sum()) + int(dec.sum())
        dices.append(1.0 if s == 0 else 2.0 * int((orig & dec).sum()) / s)

    # The other direction, and the one this script originally MISSED. Iterating the CSV
    # only ever inspects images the CSV mentions, so an image whose rows were all dropped
    # is invisible -- in a tool whose entire purpose is catching silent loss. An image
    # absent from the submission is scored as a prediction of nothing, so if it held real
    # foreground that is pure recall thrown away.
    # An absent image whose foreground is ONLY components below min_area is the instance filter
    # emptying it (rule 4 explains every pixel), not silent loss: counted separately, not failed.
    # One component >= min_area makes it absent_bad.
    absent_ok, absent_bad, absent_small = [], [], []
    for p in sorted(pred_dir.glob("*.png")):
        image_id = p.stem
        if image_id in per_image:
            continue
        fg = np.asarray(Image.open(p)) > 0
        if not fg.any():
            absent_ok.append(image_id)
            continue
        lab = ndimage.label(fg)[0]  # type: ignore[index]
        sizes = np.bincount(np.asarray(lab).ravel())[1:].tolist()
        if max(sizes) < min_area:
            absent_small.append(image_id)
            dropped_sizes.extend(sizes)
        else:
            absent_bad.append(image_id)
    if absent_bad:
        fail.append(f"{len(absent_bad)} images have foreground in the prediction but NO "
                    f"rows in the CSV, so the submission silently predicts nothing for "
                    f"them: {absent_bad[:4]}")

    if missing:
        fail.append(f"{len(missing)} image ids in the CSV have no prediction PNG: "
                    f"{missing[:4]}")
    if invented:
        fail.append(f"the submission invents {invented} foreground pixels not present in "
                    f"the predictions — the encoder is fabricating structure")
    big = [s for s in dropped_sizes if s >= min_area]
    if big:
        fail.append(f"{len(big)} dropped components are >= min_area={min_area} "
                    f"(largest {max(big)} px), so they are NOT explained by instance "
                    f"filtering — suspect a geometric encoding fault")

    # ⛔ DISJOINTNESS. Submitted rows must be the connected components of ONE binary mask
    # (`LOOP.md` rule 1), so no two rows of an image may share a pixel. Overlap is the signature of
    # the duplication exploit the host's evaluator is vulnerable to and which this project has
    # reported and refuses to use. Until 2026-08-15 nothing checked it, and a 5-row all-duplicate
    # CSV passed this gate at Dice 1.000000.
    if overlaps:
        px = sum(o[2] for o in overlaps)
        fail.append(f"{len(overlaps)} row pairs overlap ({px} shared pixels) — rows are NOT the "
                    f"connected components of one mask. Overlapping rows are the duplication "
                    f"shape the competition rules rule out; refusing.")

    report = {
        "csv": str(csv_path), "pred_dir": str(pred_dir),
        "rows": len(ids), "images": len(per_image), "cr_bytes": n_cr,
        "pred_pngs": len(list(pred_dir.glob("*.png"))),
        "absent_from_csv_empty": len(absent_ok),
        "absent_from_csv_with_foreground": len(absent_bad),
        "absent_from_csv_below_min_area": len(absent_small),
        "absent_below_min_area_ids": absent_small,
        "invented_pixels": invented,
        "dropped_components": len(dropped_sizes),
        "dropped_pixels": int(sum(dropped_sizes)),
        "largest_dropped_component": int(max(dropped_sizes)) if dropped_sizes else 0,
        "min_area": min_area,
        "overlapping_row_pairs": len(overlaps or []),
        "overlapping_pixels": int(sum(o[2] for o in (overlaps or []))),
        "dice_decoded_vs_predictions_mean": float(np.mean(dices)) if dices else None,
        "dice_decoded_vs_predictions_min": float(min(dices)) if dices else None,
        "failures": fail,
    }
    return report, fail, absent_ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_path")
    ap.add_argument("pred_dir", help="the by_image/ PNG directory the CSV was built from")
    ap.add_argument("--min-area", type=int, default=20,
                    help="must match the value make_submission.py was run with")
    ap.add_argument("--json", default="", help="write the report here")
    a = ap.parse_args(argv)

    n_cr = Path(a.csv_path).read_bytes().count(b"\r")
    per_image, problems, ids, overlaps = decode_rows(a.csv_path)
    report, fail, absent_ok = check(per_image, ids, problems, n_cr, Path(a.pred_dir),
                                   a.min_area, a.csv_path, overlaps=overlaps)

    print(f"rows {report['rows']}   images {report['images']}/{report['pred_pngs']} "
          f"prediction PNGs   CR bytes {n_cr}")
    if absent_ok:
        print(f"absent from CSV but genuinely empty (fine): {len(absent_ok)} "
              f"{absent_ok[:3]}")
    mean_dice = report["dice_decoded_vs_predictions_mean"]
    if mean_dice is not None:
        print(f"decoded-vs-predictions pixel Dice: mean {mean_dice:.6f}  "
              f"min {report['dice_decoded_vs_predictions_min']:.6f}")
    print(f"invented pixels {report['invented_pixels']}   dropped "
          f"{report['dropped_components']} components / {report['dropped_pixels']} px, "
          f"largest {report['largest_dropped_component']} (< min_area={a.min_area})")
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1))
        print(f"wrote {a.json}")
    if fail:
        # Capped: a per-row failure mode (e.g. every RLE undecodable) produces one entry
        # per row, and printing 2,000 near-identical lines buries the summary above them.
        # The full list always reaches --json.
        print(f"\nFAILED ({len(fail)} problem{'s' if len(fail) != 1 else ''}):")
        for f in fail[:15]:
            print(f"  - {f}")
        if len(fail) > 15:
            print(f"  … and {len(fail) - 15} more"
                  + (f" (full list in {a.json})" if a.json else
                     " — pass --json to capture them all"))
        sys.exit(1)
    print("\nOK — the submission decodes to the predictions it was built from.")


if __name__ == "__main__":
    raise SystemExit(main())
