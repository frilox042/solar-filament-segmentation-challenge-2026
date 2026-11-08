#!/usr/bin/env python3
"""Morphology of the kept OOF masks (width, branching, split degree, barb coverage) on the 707 observations.

    python scripts/eval_morphometry.py [--workers 3] [--out JSON]

Needs WORK staged with `tests/smoke/stage_work.py --eval` and the cross-fitted keeps
(`scripts/eval_oof_selection.py`). Method: `filament_seg.evaluation.morphometry`.
GATE: every value in configs/eval/expected.json `morphometry` (width, branching, split, selector deletion, barbs) is
reproduced to 1e-9; every rendered candidate set equals the scored OOF rows (asserted per image).
NEGATIVE CONTROLS: (i) the D007 arm at a 100 px cut must give a different width excess than at 400 px (and the reported
value); (ii) the base at threshold 0.92 instead of 0.97 must give a different candidate set on at least one of the
sampled images (the area gate can fail).
CPU only; ~3 GiB with 3 workers.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filament_seg import config as C  # noqa: E402
from filament_seg import filament_paths as fp  # noqa: E402
from filament_seg.evaluation import expected, oof_selection  # noqa: E402
from filament_seg.evaluation import morphometry as MO  # noqa: E402

D007_DIR = "predictions/vessel_oof_full/by_image"     # relative to WORK (configs/eval/inputs.json)


def gate(res: dict, exp: dict) -> dict:
    tol = exp["tol"]
    ch = {}
    for sec in ("width", "branching", "split", "selector_deletion_bag0", "barbs"):
        ch.update(expected.compare_tree(res[sec], exp[sec], tol, sec))
    w100 = res["width"]["D007_100"]["signed_px"]
    ch["negative_control_width_cut100"] = {
        **expected.close(w100, exp["negative_control_width_D007_100_signed_px"], tol),
        "differs_from_cut400": bool(abs(w100 - res["width"]["D007_400"]["signed_px"]) > 1e-3)}
    ch["negative_control_width_cut100"]["pass"] &= ch["negative_control_width_cut100"]["differs_from_cut400"]
    nc = res["negative_control_threshold_candidates_equal"]
    ch["negative_control_threshold"] = {"sampled": len(nc), "equal": sum(nc), "pass": bool(nc) and not all(nc)}
    return {"pass": all(v["pass"] for v in ch.values()), "failed": [k for k, v in ch.items() if not v["pass"]],
            "checks": ch}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--expect", help="model set in configs/eval/expected.json (default: its `current`)")
    ap.add_argument("--out", help="result JSON (default WORK/eval/morphometry.json)")
    a = ap.parse_args()
    exp = expected.load("morphometry", a.expect)
    t0 = time.time()
    res = MO.run({"FAST": C.load("FAST")}, str(fp.WORK / D007_DIR), a.workers)
    out = {"what": "morphology of the kept OOF masks", "results": res, "gate": gate(res, exp),
           "seconds": round(time.time() - t0, 1)}
    path = Path(a.out) if a.out else oof_selection.eval_dir() / "morphometry.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, default=float))
    w, b, s, br = res["width"], res["branching"], res["split"], res["barbs"]
    print(f"width excess (% of {w['denominator_px']:.2f} px): D007 {w['D007_400']['pct']:+.1f}  "
          f"FAST {w['FAST']['pct']:+.1f}; abs error {w['D007_400']['abs_px']:.2f} -> {w['FAST']['abs_px']:.2f} px")
    print(f"branching vs reference (n {b['n_pop']}): D007 {b['D007_all']['pct']:+.1f}%  FAST {b['FAST']['pct']:+.1f}% "
          f"[{b['FAST']['pct_ci'][0]:+.1f}, {b['FAST']['pct_ci'][1]:+.1f}]")
    print(f"split share: D007 {s['D007_400']:.4f}  FAST {s['FAST']:.4f}; FAST coverage of split "
          f"annotations {res['selector_deletion_bag0']['FAST']['mean_coverage_candidates']:.2f} -> "
          f"{res['selector_deletion_bag0']['FAST']['mean_coverage_kept']:.2f}")
    cov = br["barb_coverage_per_image_mean"]
    print(f"barb px outside & p < 0.3: D007 {br['outside_p_lt_0.3']['D007_all']:.3f}; barb coverage FAST "
          f"{cov['FAST']:.3f} vs D007 {cov['D007_400']:.3f}")
    print("gate", "PASS" if out["gate"]["pass"] else f"FAIL {out['gate']['failed']}", f"{out['seconds']} s")
    return 0 if out["gate"]["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
