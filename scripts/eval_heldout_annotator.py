#!/usr/bin/env python3
"""Paper Table 1: the models against a held-out annotator on the host's metric (296 OOF observations).

    python scripts/eval_heldout_annotator.py [--workers 3] [--out JSON]

Needs WORK staged with `tests/smoke/stage_work.py --eval` and the cross-fitted keeps
(`scripts/eval_oof_selection.py`, cached in WORK/eval/). Method: `filament_seg.evaluation.heldout_annotator`.

Two bootstrap seeds are run on the same counts: 20261018 (the seed of the reported intervals) and 0 (the seed of the
second reported set). GATE: every point estimate, delta and interval listed in
configs/eval/expected.json `heldout_annotator` is reproduced (1e-9; 1e-6 for the float32 Dice of the second set).
NEGATIVE CONTROL: putting the held-out record back into its own reference must raise the human's PQ by > 0.2
(a broken exclusion is detectable), and equal the reported self-reference value.
CPU only; ~3 GiB with 3 workers; about 15 min.
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
from filament_seg.evaluation import heldout_annotator as HA  # noqa: E402

D007_DIR = "predictions/vessel_oof_full/by_image"     # relative to WORK (configs/eval/inputs.json)


def gate(res: dict, exp: dict) -> dict:
    tol = exp["tol"]
    A, P = res[f"seed_{exp['rerun_seed']}"], res[f"seed_{exp['reported_seed']}"]
    checks = {"n_images": A["n_images"] == exp["n_images"], "n_units": A["n_units"] == exp["n_units"]}
    for arm, st in exp["points"].items():
        checks[f"point.{arm}"] = expected.close([A["points"][arm][s] for s in st], list(st.values()), tol)
    for src, R in (("rerun", A), ("reported", P)):
        for key, st in exp[src].items():
            if key in ("dice", "far"):
                continue
            mine = [[R["pooled"][key][s]["delta"], *R["pooled"][key][s]["ci95"]] for s in st]
            checks[f"{src}.{key}"] = expected.close(mine, [[v["delta"], *v["ci95"]] for v in st.values()], tol)
    d = exp["rerun"]["dice"]
    for m in ("H", "nnHALO", "D007"):
        checks[f"rerun.dice.{m}_image_mean"] = expected.close(A["dice"][f"{m}_image_mean"], d[f"{m}_image_mean"], tol)
    for m in ("nnHALO", "D007"):
        for src, R, t in (("rerun", A, tol), ("reported", P, exp["tol_reported_dice"])):
            v = exp[src]["dice"][f"{m}_minus_H"]
            mine = R["dice"][f"{m}_minus_H"]
            checks[f"{src}.dice.{m}_minus_H"] = expected.close([mine["mean_diff"], *mine["ci95"]],
                                                              [v["mean_diff"], *v["ci95"]], t)
    f = exp["rerun"]["far"]
    checks["rerun.far.n"] = A["far"]["n"] == f["n"]
    for m in ("nnHALO",):
        for s, v in f[f"{m}_vs_H_A0"].items():
            mine = A["far"][f"{m}_vs_H_A0"][s]
            checks[f"rerun.far.{m}.{s}"] = expected.close([mine["delta"], *mine["ci95"]], [v["delta"], *v["ci95"]], tol)
    nc = A["negative_control_selfref"]
    checks["negative_control_selfref"] = {**nc, **expected.close(nc["pq_w05_selfref"],
                                                                 exp["negative_control_selfref_pq_w05"], tol)}
    checks["negative_control_selfref"]["pass"] = bool(nc["pass"] and checks["negative_control_selfref"]["pass"])
    ok = all(v["pass"] if isinstance(v, dict) else bool(v) for v in checks.values())
    failed = [k for k, v in checks.items() if not (v["pass"] if isinstance(v, dict) else v)]
    return {"pass": ok, "failed": failed, "checks": checks}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--expect", help="model set in configs/eval/expected.json (default: its `current`)")
    ap.add_argument("--out", help="result JSON (default WORK/eval/heldout_annotator.json)")
    a = ap.parse_args()
    exp = expected.load("heldout_annotator", a.expect)
    t0 = time.time()
    res = HA.run({"nnHALO": C.load("nnHALO")}, str(fp.WORK / D007_DIR), a.workers,
                 (exp["rerun_seed"], exp["reported_seed"]))
    out = {"what": "held-out annotator vs models (paper Table 1)", "results": res, "gate": gate(res, exp),
           "seconds": round(time.time() - t0, 1)}
    path = Path(a.out) if a.out else oof_selection.eval_dir() / "heldout_annotator.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, default=float))
    A = res[f"seed_{exp['rerun_seed']}"]
    for m in ("nnHALO", "D007"):
        c = A["pooled"][f"{m}_vs_H_A0"]
        print(f"{m:5s} PQ {c['pq_w05']['model']:.4f} vs annotator {c['pq_w05']['reference']:.4f}  "
              f"delta {c['pq_w05']['delta']:+.4f} [{c['pq_w05']['ci95'][0]:+.4f}, {c['pq_w05']['ci95'][1]:+.4f}]  "
              f"Dice {A['dice'][f'{m}_image_mean']:.4f} ({A['dice'][f'{m}_minus_H']['mean_diff']:+.4f})  "
              f"IoU {c['mean_iou']['model']:.4f}")
    print("gate", "PASS" if out["gate"]["pass"] else f"FAIL {out['gate']['failed']}", f"{out['seconds']} s")
    return 0 if out["gate"]["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
