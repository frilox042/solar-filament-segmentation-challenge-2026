#!/usr/bin/env python3
"""Gate for `scripts/build_labels.py`: the built labels equal the shipped members' training labels on 707 / 707.

    python tests/eval_records/build_labels_check.py --reference-raw <nnUNet_raw of the shipped members> --out OUT.json [--workers 4]

1. BUILD   `filament_seg.prep.labels.build` writes every label of D005/D007/D012/D036 into $nnUNet_raw (must be a
           scratch tree; refused if it is the reference). The WRITTEN files are then read back and compared value for
           value with `<reference>/<Dataset>/labelsTr/<case>.png` (D007/D012 reference labels are links to D005's).
           PNG byte identity is recorded too, but it is not the gate.
2. NEGATIVE CONTROLS, in memory, through the same rebuild path, on both the union and the barb label:
   drop    the first record of the first multi-record observation removed: exactly that case must break.
   shift   every polygon x coordinate + 1 px before pycocotools: every case must break.
Needs: the competition download ($FILAMENT_KAGGLE_ROOT, $FILAMENT_KAGGLE_COCO) and the reference tree. CPU only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from filament_seg import filament_paths as fp  # noqa: E402
from filament_seg.evaluation import records_coco as RC  # noqa: E402
from filament_seg.prep import labels as L  # noqa: E402


def _arr(p: Path) -> np.ndarray:
    a = np.asarray(Image.open(p))
    return a[..., 0] if a.ndim == 3 else a


def compare_written(raw: Path, ref: Path, cases) -> dict:
    out = {}
    for ds in L.DATASETS.values():
        n_eq = n_bytes = 0
        max_diff = 0
        bad = []
        for c in cases:
            a, b = raw / ds / "labelsTr" / f"{c}.png", ref / ds / "labelsTr" / f"{c}.png"
            x, y = _arr(a), _arr(b)
            eq = x.shape == y.shape and x.dtype == y.dtype and np.array_equal(x, y)
            n_eq += eq
            max_diff = max(max_diff, int((x != y).sum()) if x.shape == y.shape else -1)
            n_bytes += hashlib.md5(a.read_bytes()).digest() == hashlib.md5(b.read_bytes()).digest()
            if not eq:
                bad.append(c)
        vals = sorted(np.unique(_arr(raw / ds / "labelsTr" / f"{cases[1]}.png")).tolist())
        out[ds] = {"n_cases": len(cases), "n_equal": n_eq, "max_diff_px": max_diff, "n_png_bytes_identical": n_bytes,
                   "values_img_0001": vals, "unequal": bad[:10]}
    return out


def shifted(recs_by_case: dict, dx: int = 1) -> dict:
    def sh(seg):
        return [[v + dx if i % 2 == 0 else v for i, v in enumerate(poly)] for poly in seg]
    return {c: [RC.Record(r.id, tuple(sh(s) for s in r.segs)) for r in rs] for c, rs in recs_by_case.items()}


def broken(res: dict, k: str) -> list:
    return sorted(c for c, r in res.items() if not r[k]["equal"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference-raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    raw, ref = fp.NNUNET_RAW.resolve(), a.reference_raw.resolve()
    if raw == ref or ref in raw.parents or raw in ref.parents:
        sys.exit(f"refusing: $nnUNet_raw ({raw}) overlaps the reference tree ({ref})")
    t0 = time.time()
    built = L.build(list(L.DATASETS), raw=raw, workers=a.workers)
    cases, recs = L.records()
    cases = sorted(cases)
    written = compare_written(raw, ref, cases)
    t_build = time.time() - t0

    nc_case = next(c for c in cases if len(recs[c]) >= 2)
    dropped = recs[nc_case][0].id
    drop = dict(recs, **{nc_case: recs[nc_case][1:]})
    res_drop = L.verify(["D005", "D036"], ref, workers=a.workers, recs_override=drop)
    res_shift = L.verify(["D005", "D036"], ref, workers=a.workers, recs_override=shifted(recs))
    controls = {
        "drop": {"case": nc_case, "dropped_record": dropped,
                 **{k: {"broken": broken(res_drop, k), "diff_px": res_drop[nc_case][k]["n_diff"]}
                    for k in ("D005", "D036")}},
        "shift_x_plus_1px": {k: {"n_broken": len(broken(res_shift, k)), "n_cases": len(res_shift),
                                 "min_diff_px": min(r[k]["n_diff"] for r in res_shift.values())}
                             for k in ("D005", "D036")},
    }
    ok_ident = all(v["n_equal"] == len(cases) == 707 and v["max_diff_px"] == 0 for v in written.values())
    ok_drop = all(controls["drop"][k]["broken"] == [nc_case] for k in ("D005", "D036"))
    ok_shift = all(v["n_broken"] == v["n_cases"] for v in controls["shift_x_plus_1px"].values())
    rec = {"what": "scripts/build_labels.py labels == shipped members' labelsTr (value for value), written files",
           "coco": fp.KAGGLE_COCO.name, "reference": "nnUNet_raw of the shipped members: {Dataset005,007,012,036}/labelsTr",
           "n_cases": len(cases), "n_records": sum(len(r) for r in recs.values()), "build": built,
           "identity": written, "negative_controls": controls, "seconds": round(time.time() - t0, 1),
           "seconds_build_and_compare": round(t_build, 1),
           "pass": {"identity": ok_ident, "drop": ok_drop, "shift": ok_shift}}
    rec["pass"]["all"] = all(rec["pass"].values())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec["pass"]))
    return 0 if rec["pass"]["all"] else 1


if __name__ == "__main__":
    sys.exit(main())
