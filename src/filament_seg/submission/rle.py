"""Build the Kaggle submission CSV: per-instance COCO-RLE rows.

Input: binary test predictions remapped to image ids (predictions/<tag>/by_image/<imageid>.png).
Output: submission CSV with columns filament_id (<imageid>_<n>) and segmentation_rle.

Encoding: pycocotools COCO RLE (compressed counts), decoded to ASCII, no quotes,
fixed 2048x2048 — matching the competition spec.

⚠️ Line endings are LOAD-BEARING, and this file used to get them wrong. `csv.writer`'s
default `excel` dialect sets `lineterminator="\r\n"`, and `open(..., newline="")` — the
documented, correct-looking spelling — deliberately suppresses translation so that CRLF
reaches the file verbatim. **Every submission before 2026-07-26 was therefore CRLF.**

Why that is a defect: a CR-suffixed RLE **does not decode**. `pycocotools` raises
`ValueError: Invalid RLE mask representation` on `PPP8\r`, on `PPPP4\r`, and on every real
counts string from a built submission. So a CRLF file scores correctly *only if* the
evaluator's CSV parsing strips the CR — an undocumented property of someone else's code that
we were silently depending on.

⚠️ It is **not** established that this cost us any leaderboard score, and it must not be
quoted that way. Evidence that looked like a 0.10 cost is refuted:
two of our CRLF submissions with identical row counts but different mask content scored
differently (0.65 vs 0.71), which is only possible if the evaluator *did* decode our rows.
The fix is correct because relying on an unspecified parser is a defect on its own terms.

A COCO RLE counts string is ASCII 48..111, so it never contains a comma, quote, CR or LF —
csv quoting never fires, which is exactly why eight submissions went out without anything
flagging this. `tests/test_rle.py` asserts LF, the round trip and non-overlapping rows.

Usage:
  python -m filament_seg.submission.rle <by_image_dir> <out_csv> [--min-area N] [--watershed]
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask
from scipy import ndimage


def separate_instances(binary, min_area=20, use_watershed=False, min_distance=25):
    """binary -> int label map (0=bg, 1..N). Instances below min_area are dropped."""
    binary = np.asarray(binary) > 0
    if not binary.any():
        return np.zeros(binary.shape, dtype=np.int32)

    if use_watershed:
        from skimage.feature import peak_local_max
        from skimage.segmentation import watershed
        dist = ndimage.distance_transform_edt(binary)
        coords = peak_local_max(dist, min_distance=min_distance, labels=binary)
        markers = np.zeros(dist.shape, dtype=np.int32)
        for i, (r, c) in enumerate(coords, start=1):
            markers[r, c] = i
        lab = watershed(-dist, markers, mask=binary) if markers.max() > 0 else ndimage.label(binary)[0]
    else:
        lab, _ = ndimage.label(binary)

    # relabel contiguously, dropping sub-min-area components
    out = np.zeros(lab.shape, dtype=np.int32)
    idx = 1
    for lbl in range(1, int(lab.max()) + 1):
        m = lab == lbl
        if m.sum() >= min_area:
            out[m] = idx
            idx += 1
    return out


def rle_for(mask_bool: np.ndarray) -> str:
    """COCO RLE counts string for a single binary instance mask (HxW bool)."""
    rle = coco_mask.encode(np.asfortranarray(mask_bool.astype(np.uint8)))
    return rle["counts"].decode("ascii")


def rows_for_mask(image_id: str, pred: np.ndarray, min_area: int = 20, watershed: bool = False) -> list:
    """The CSV rows of one image: (f"{image_id}_{n}", RLE counts) for each instance >= min_area px."""
    lab = separate_instances(pred, min_area=min_area, use_watershed=watershed)
    return [(f"{image_id}_{n}", rle_for(lab == n)) for n in range(1, int(lab.max()) + 1)]


def write_csv(by_image_dir, out_csv, min_area: int = 20, watershed: bool = False) -> dict:
    """One row per instance of every <image_id>.png in by_image_dir (sorted): filament_id = <image_id>_<n>."""
    imgs = sorted(glob.glob(os.path.join(str(by_image_dir), "*.png")))
    if not imgs:
        sys.exit(f"no PNGs in {by_image_dir}")
    rows, n_inst, n_empty = [], 0, 0
    for p in imgs:
        image_id = os.path.splitext(os.path.basename(p))[0]
        r = rows_for_mask(image_id, np.asarray(Image.open(p)) > 0, min_area, watershed)
        if not r:
            n_empty += 1
        rows += r
        n_inst += len(r)
    with open(out_csv, "w", newline="") as f:
        # lineterminator="\n" is REQUIRED, not stylistic -- see the module docstring.
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["filament_id", "segmentation_rle"])
        w.writerows(rows)
    print(f"wrote {out_csv}: {len(imgs)} images, {n_inst} filament instances "
          f"({n_empty} images with none). avg {n_inst/len(imgs):.1f}/image")
    return {"images": len(imgs), "instances": n_inst, "empty": n_empty}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("by_image_dir")
    ap.add_argument("out_csv")
    ap.add_argument("--min-area", type=int, default=20)
    ap.add_argument("--watershed", action="store_true")
    a = ap.parse_args(argv)
    write_csv(a.by_image_dir, a.out_csv, a.min_area, a.watershed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
