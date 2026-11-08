"""Agreement of a submission CSV with a reference CSV (the uploaded file), judged by one reproduction rule.

The GPU stage is not deterministic run to run (two runs of the same environment differ by max |dp| up to 1.3e-3), and
a few candidates score within about 1e-3 of the selector's cut, so a GPU re-run can keep or delete one of them. A
re-run is therefore judged by AGREEMENT, not by md5. Each image is compared by its object count (rows) and by its
union mask. The rule (`configs/FAST.json` `reproduction_tolerance.rule`) PASSES iff

  at most 3 images whose object count differs, each by at most 1 object,
  AND the total object count within +-3 of the reference,
  AND the pooled union-mask Dice over the images whose object count MATCHES (2 sum|A & B| / sum(|A| + |B|))
      >= 0.999.

A borderline candidate kept in one run and deleted in the other changes the count of its image; the count clauses
bound how many such flips are allowed, and the Dice clause checks that everything else agrees.

(`verify_submission` must pass as well; `selector_stage` checks that.) Reported as information, never deciding:
`md5_matches_reference` (bit-identical file), `same_count_everywhere` (object count equal on every image),
`pooled_dice_all` (over all images), `min_image_dice` / `worst_image` (all images) and `min_matched_image_dice` /
`worst_matched_image` (count-matched images). An image id outside the test set, or a test image the comparison did
not cover, fails the rule.

    python -m filament_seg.submission.compare OUT.csv REFERENCE.csv [--slot FAST] [--json OUT.json]
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict

import numpy as np

SIZE = 2048
RULE_KEYS = ("max_images_count_differs", "max_count_diff_per_image", "max_total_count_diff",
             "min_pooled_dice_count_matched")
def _rows(path) -> dict:
    """image id -> list of RLE counts strings."""
    csv.field_size_limit(1 << 30)
    out: dict = defaultdict(list)
    with open(path, newline="") as f:
        r = csv.reader(f)
        next(r, None)
        for fid, rle in r:
            out[fid.rsplit("_", 1)[0]].append(rle)
    return out


def _union(rles: list) -> np.ndarray:
    from pycocotools import mask as coco_mask
    m = np.zeros((SIZE, SIZE), bool)
    for c in rles:
        m |= coco_mask.decode({"size": [SIZE, SIZE], "counts": c.encode("ascii")}).astype(bool)
    return m


def failures(rule: dict, count_diff: dict, rows: int, reference_rows: int, pooled_dice: float | None,
             structural: list = ()) -> list:
    """Why the rule fails (empty list: it passes). `count_diff`: image -> rows minus reference rows, non-zero only;
    `structural`: reasons that fail it regardless (unknown or uncovered image ids)."""
    out = list(structural)
    n, worst_d = len(count_diff), max((abs(v) for v in count_diff.values()), default=0)
    if n > rule["max_images_count_differs"]:
        out.append(f"object count differs on {n} images (allowed {rule['max_images_count_differs']})")
    if worst_d > rule["max_count_diff_per_image"]:
        out.append(f"an image's object count differs by {worst_d} (allowed {rule['max_count_diff_per_image']})")
    if abs(rows - reference_rows) > rule["max_total_count_diff"]:
        out.append(f"total object count {rows} vs {reference_rows} (allowed +-{rule['max_total_count_diff']})")
    if pooled_dice is None or pooled_dice < rule["min_pooled_dice_count_matched"]:
        out.append(f"pooled Dice over count-matched images {pooled_dice} < {rule['min_pooled_dice_count_matched']}")
    return out


def classify(rule: dict, count_diff: dict, rows: int, reference_rows: int, pooled_dice: float | None,
             structural: list = ()) -> dict:
    """{within_tolerance, failures, same_count_everywhere} for plain numbers; `pooled_dice` is the count-matched one."""
    f = failures(rule, count_diff, rows, reference_rows, pooled_dice, structural)
    return {"within_tolerance": not f, "failures": f, "same_count_everywhere": not count_diff and not structural}


def compare(csv_path, reference_path, image_ids, rule: dict) -> dict:
    """Per-image count and union-mask agreement of `csv_path` with `reference_path` over `image_ids` (all test
    images), judged by `rule` (`reproduction_tolerance.rule` of the slot config)."""
    assert set(rule) == set(RULE_KEYS), sorted(rule)
    a, b = _rows(csv_path), _rows(reference_path)
    unknown = sorted((set(a) | set(b)) - set(image_ids))
    count_diff, dice = {}, {}
    inter = total = inter_m = total_m = 0
    for i in image_ids:
        ra, rb = a.get(i, []), b.get(i, [])
        if len(ra) != len(rb):
            count_diff[i] = len(ra) - len(rb)
        ma = _union(ra)
        mb = ma if ra == rb else _union(rb)
        s, x = int(ma.sum()) + int(mb.sum()), int((ma & mb).sum())
        inter, total = inter + x, total + s
        if len(ra) == len(rb):
            inter_m, total_m = inter_m + x, total_m + s
        dice[i] = 1.0 if s == 0 else 2.0 * x / s
    worst = min(dice, key=dice.get) if dice else None
    matched = {i: d for i, d in dice.items() if i not in count_diff}
    worst_m = min(matched, key=matched.get) if matched else None
    pooled_all = (1.0 if total == 0 else 2.0 * inter / total) if dice else None
    pooled = (1.0 if total_m == 0 else 2.0 * inter_m / total_m) if matched else None
    rows, ref_rows = sum(map(len, a.values())), sum(map(len, b.values()))
    structural = [f"image ids not in the test set: {unknown}"] if unknown else []
    if len(dice) != len(image_ids) or not image_ids:
        structural.append(f"{len(dice)} images compared, {len(image_ids)} expected")
    verdict = classify(rule, count_diff, rows, ref_rows, pooled, structural)
    md5 = hashlib.md5(open(csv_path, "rb").read()).hexdigest()
    md5_ref = hashlib.md5(open(reference_path, "rb").read()).hexdigest()
    return {"reference": str(reference_path), "n_images": len(image_ids), "rows": rows, "reference_rows": ref_rows,
            "images_count_differs": count_diff, "pooled_dice_count_matched": pooled, "pooled_dice_all": pooled_all,
            "min_matched_image_dice": matched[worst_m] if worst_m else None, "worst_matched_image": worst_m,
            "min_image_dice": dice[worst] if worst else None, "worst_image": worst, "unknown_image_ids": unknown,
            "images_identical_rows": sum(a.get(i, []) == b.get(i, []) for i in image_ids),
            **verdict, "md5": md5, "reference_md5": md5_ref, "md5_matches_reference": md5 == md5_ref}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("reference")
    ap.add_argument("--slot", default="FAST", choices=("FAST",))
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    from filament_seg import config as C
    cfg = C.load(a.slot)
    ids = sorted(C.test_image_ids(cfg))
    res = compare(a.csv, a.reference, ids, cfg.tolerance_rule)
    print(json.dumps(res, indent=1))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
    return 0 if res["within_tolerance"] else 1


if __name__ == "__main__":
    sys.exit(main())
