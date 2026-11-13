#!/usr/bin/env python3
"""Calibration (ECE) of the selector score on the 707 OOF observations, and its scale on test.

    python scripts/eval_calibration.py [--test-bag PKL] [--out JSON]

Needs the cross-fitted keeps (`scripts/eval_oof_selection.py`). Method: `filament_seg.evaluation.calibration`.
GATE: bag-mean raw ECE, cross-fitted isotonic ECE, their bootstrap CIs, the per-bag values, both estimator controls
and the pooled OOF mean score equal configs/eval/expected.json `calibration` (1e-12).
NEGATIVE CONTROL: the shifted score s + 0.2 must read as miscalibrated (ECE > 0.05) while synthetic labels drawn from s
read as calibrated (< 0.03).
--test-bag: also the test side (mean raw score of the kept test objects under a bag, e.g. the released
weights/selector/nnHALO_bag.pkl; needs the test network outputs in WORK, e.g. staged by tests/smoke/stage_work.py).
No test annotation is read; the test side is reported, not gated.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filament_seg import config as C  # noqa: E402
from filament_seg.evaluation import calibration, expected, oof_selection  # noqa: E402


def gate(res: dict, exp: dict) -> dict:
    t = exp["tol"]
    ch = {
        "ece_raw": expected.close([res["ece_raw_bagmean"], *res["ece_raw_ci95"]],
                                  [exp["ece_raw_bagmean"], *exp["ece_raw_ci95"]], t),
        "ece_isotonic_oof": expected.close([res["ece_isotonic_oof_bagmean"], *res["ece_isotonic_oof_ci95"]],
                                           [exp["ece_isotonic_oof_bagmean"], *exp["ece_isotonic_oof_ci95"]], t),
        "per_bag": expected.close([[p["ece_raw"], p["ece_isotonic_oof"], p["n_kept"]] for p in res["per_bag"]],
                                  [list(x) for x in zip(exp["per_bag_ece_raw"], exp["per_bag_ece_isotonic_oof"],
                                                        exp["per_bag_n_kept"], strict=True)], t),
        "controls": expected.close([res["controls"]["calibrated_synthetic_y_ece"],
                                    res["controls"]["shifted_s_plus_0.2_ece"]],
                                   [exp["controls"]["calibrated_synthetic_y_ece"],
                                    exp["controls"]["shifted_s_plus_0.2_ece"]], t),
        "negative_control_shift_detected": {"pass": res["controls"]["pass"]},
        "oof_pooled_mean": expected.close([res["oof_kept_pooled"]["n"], res["oof_kept_pooled"]["mean_raw_score"]],
                                          [exp["oof_kept_pooled_n"], exp["oof_kept_pooled_mean_raw_score"]], t),
    }
    return {"pass": all(v["pass"] for v in ch.values()), "failed": [k for k, v in ch.items() if not v["pass"]],
            "checks": ch}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slot", default="nnHALO", choices=("nnHALO",))
    ap.add_argument("--test-bag", help="bag pickle of the deployed selector (enables the test side)")
    ap.add_argument("--expect", help="model set in configs/eval/expected.json (default: its `current`)")
    ap.add_argument("--out", help="result JSON (default WORK/eval/calibration_<slot>.json)")
    a = ap.parse_args()
    exp = expected.load("calibration", a.expect)
    t0 = time.time()
    cfg = C.load(a.slot)
    res = calibration.run(oof_selection.build(cfg))
    if a.test_bag:
        d = pickle.load(open(a.test_bag, "rb"))
        res["test"] = calibration.test_scale(cfg, d["bag"], d["cut"])
    out = {"what": f"selector-score calibration, {a.slot}", "results": res,
           "gate": gate(res, exp) if a.slot == exp["slot"] else {"pass": True, "note": "no published values"},
           "seconds": round(time.time() - t0, 1)}
    path = Path(a.out) if a.out else oof_selection.eval_dir() / f"calibration_{a.slot}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, default=float))
    print(f"ECE raw {res['ece_raw_bagmean']:.4f} {res['ece_raw_ci95']}  isotonic (cross-fitted) "
          f"{res['ece_isotonic_oof_bagmean']:.4f} {res['ece_isotonic_oof_ci95']}  OOF kept mean score "
          f"{res['oof_kept_pooled']['mean_raw_score']:.4f}"
          + (f"  test kept mean {res['test']['mean_raw_score']:.4f} (n {res['test']['n_kept']})"
             if "test" in res else ""))
    print("gate", "PASS" if out["gate"]["pass"] else f"FAIL {out['gate']['failed']}", f"{out['seconds']} s")
    return 0 if out["gate"]["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
