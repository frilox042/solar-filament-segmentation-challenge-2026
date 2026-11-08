#!/usr/bin/env python3
"""Check: `records_png.union_mask` reproduces the `data/competition_labels` tree on 707 / 707 cases.

    python tests/eval_records/records_png_union_check.py --extracted <extracted/training> \
        --reference <data/competition_labels> --out OUT.json

`data/competition_labels/<case>.png` (uint8, 0/1) is an earlier union-target tree built from the PNG export: the OR of
every class PNG of every record of the `extracted/` tree, which `records_png.union_mask` computes.
Compared value for value. NEGATIVE CONTROL: the first record of the first multi-record case is dropped; exactly that
case must break. Needs the historical `extracted/` tree (not in the competition download). CPU only, read-only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

EXPORT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXPORT / "src"))
from filament_seg.evaluation import records_png as RP  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extracted", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--case-map", type=Path, default=EXPORT / "configs" / "case_mapping_full.json")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    by = RP.records_by_case(a.case_map, a.extracted)
    diffs = {}
    for c in sorted(by):
        ref = np.asarray(Image.open(a.reference / f"{c}.png"))
        diffs[c] = int((RP.union_mask(by[c]).astype(np.uint8) != ref).sum())
    ctl_case = next(c for c in sorted(by) if len(by[c]) > 1)
    ctl_broken = [c for c in sorted(by) if
                  (RP.union_mask(by[c][1:] if c == ctl_case else by[c]).astype(np.uint8)
                   != np.asarray(Image.open(a.reference / f"{c}.png"))).any()]
    res = {"n_cases": len(diffs), "n_equal": sum(d == 0 for d in diffs.values()),
           "max_diff_px": max(diffs.values()),
           "negative_control": {"dropped_first_record_of": ctl_case, "broken_cases": ctl_broken,
                                "PASS": ctl_broken == [ctl_case]},
           "seconds": round(time.time() - t0)}
    res["PASS"] = res["n_equal"] == res["n_cases"] == 707 and res["negative_control"]["PASS"]
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res))
    return 0 if res["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
