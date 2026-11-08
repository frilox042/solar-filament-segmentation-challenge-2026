#!/usr/bin/env python3
"""The never-seen class: the annotated filaments a model never fires on are faint AND single-annotator.

    python scripts/eval_never_seen.py [--records coco|png] [--workers 3] [--out JSON]
                                      [--reference-rows JSON --reference-features NPY]     (png only)

Needs WORK staged with `tests/smoke/stage_work.py --eval` (OOF maps, D007 binaries, the competition COCO, the
per-frame limb fit; the historical per-record PNG tree only for --records png) and the cross-fitted keeps
(`scripts/eval_oof_selection.py`). Method: `filament_seg.evaluation.never_seen`. Writes the per-object rows
to WORK/eval/never_seen_rows.npz (never_seen_rows_png.npz for --records png), tagged with the source.
--records coco (default) gates against this code's own COCO run (block `by_records.coco` of configs/eval/expected.json:
regression values produced by this code); --records png replicates the reported values below.
GATE (png; 1e-12, counts exact): the D007 class counts D / A / B / M / C (515 / 746 / 926 / 3478 / 62); every class-vs-M
AUC, interval, median and n (contrast D 0.884, support D 0.858, ...); the support headline (8 of 284 drawn by
every record vs M 78.5%, by k); per bag of FAST the class counts, the D contrast / support AUC and interval, and
the bag means.
NOT REPRODUCED, reported only: M's all-record share reweighted to D's area distribution (reported 50.2%, this code 0.5026;
the binning of the reported value is unrecorded), see `not_reproduced` in configs/eval/expected.json.
--reference-rows / --reference-features: also check every per-object row against the original row files.
NEGATIVE CONTROLS: a 50% support rule must give the block's count (png: 7, reported) of drawn-by-every-record objects,
and differ from the 20% count (png: 8); classes shuffled within each image must give a D-vs-M contrast AUC below 0.6.
CPU only; ~3 GiB with 3 workers; about 13 min.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filament_seg import config as C  # noqa: E402
from filament_seg import filament_paths as fp  # noqa: E402
from filament_seg.evaluation import expected, oof_selection  # noqa: E402
from filament_seg.evaluation import never_seen as NS  # noqa: E402
from filament_seg.evaluation import records as REC  # noqa: E402

D007_BIN = "predictions/vessel_oof_full/by_image"     # relative to WORK (configs/eval/inputs.json)
D007_PROB = "predictions/vessel_oof_probs"


def reference_rows(rows: dict, rows_json: str, features_npy: str) -> dict:
    """Max |difference| of every per-object row against the original per-object JSON rows and feature rows."""
    F, cases = rows["F"], rows["cases"]
    a = json.load(open(rows_json))["rows"]
    f = list(np.load(features_npy, allow_pickle=True))
    out = {"n": [len(a), len(f), len(F["area"])], "stems_equal": [r["stem"] for r in a] == [r["stem"] for r in f]}
    order = [r["stem"] for r in a]
    out["case_order_equal"] = sorted(set(order)) == cases and order == sorted(order)
    for k in ("area", "iou", "p_max", "contrast", "matched", "touched_by_deleted"):
        out[k] = float(np.nanmax(np.abs(np.array([r[k] for r in a], float) - F[k].astype(float))))
    for k in ("clutter", "interior_texture", "width", "rR", "support"):
        A, B = np.array([r[k] for r in f], float), F[k].astype(float)
        out[k] = float(np.nanmax(np.abs(A - B))) if np.array_equal(np.isnan(A), np.isnan(B)) else float("inf")
    out["pass"] = bool(out["stems_equal"] and out["case_order_equal"] and len(set(out["n"])) == 1
                       and all(out[k] == 0.0 for k in ("area", "matched", "touched_by_deleted"))
                       and all(out[k] <= 1e-12 for k in ("iou", "p_max", "contrast", "clutter", "interior_texture",
                                                          "width", "rR", "support")))
    return out


def gate(res: dict, exp: dict, ref: dict | None) -> dict:
    t = exp["tol"]
    ch = {"n_objects": {"pass": res["n_objects"] == exp["n_objects"], "mine": res["n_objects"]},
          "class_counts": {"pass": res["class_counts"] == exp["class_counts"], "mine": res["class_counts"]}}
    ch.update(expected.compare_tree(res["results"], exp["results"], t, "results"))
    ch.update(expected.compare_tree(res["headline"], exp["headline"], t, "headline"))
    for m, pe in exp["per_model"].items():
        pm = res["per_model"][m]
        for k in ("contrast_AUC_bag_mean", "support_AUC_bag_mean", "contrast_AUC_bag_range", "support_AUC_bag_range",
                  "n_D_bag_mean"):
            ch[f"{m}.{k}"] = expected.close(pm[k], pe[k], t)
        for b, be in enumerate(pe["bags"]):
            mb = pm["bags"][b]
            ch[f"{m}.b{b}.class_counts"] = {"pass": mb["class_counts"] == be["class_counts"],
                                            "mine": mb["class_counts"]}
            for ft in ("contrast", "support"):
                ch[f"{m}.b{b}.D_{ft}"] = expected.close([mb["D"][ft]["auc"], *mb["D"][ft]["ci"]], be[f"D_{ft}"], t)
            ch[f"{m}.b{b}.D_drawn_by_every_record"] = {"pass": mb["D_drawn_by_every_record"]
                                                       == be["D_drawn_by_every_record"],
                                                       "mine": mb["D_drawn_by_every_record"]}
    nc = res["negative_controls"]
    n50 = nc["support_50pct"]["D_all_agree"]
    ch["negative_control_support_50pct"] = {"mine": n50, "published": exp["negative_control_support_50pct_D_all_agree"],
                                            "pass": n50 == exp["negative_control_support_50pct_D_all_agree"]
                                            and n50 != exp["headline"]["D_all_agree"]}
    a = nc["shuffled_classes_D_contrast_auc"]
    lim = exp["negative_control_shuffled_contrast_auc_below"]
    ch["negative_control_shuffled_classes"] = {"auc": a, "pass": bool(max(a, 1 - a) < lim)}
    if ref is not None:
        ch["reference_rows"] = ref
    return {"pass": all(v["pass"] for v in ch.values()), "failed": [k for k, v in ch.items() if not v["pass"]],
            "n_checks": len(ch), "max_abs": max((v.get("max_abs", 0.0) for v in ch.values()), default=0.0),
            "checks": ch}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=3)
    REC.add_argument(ap)
    ap.add_argument("--reference-rows", help="original per-object rows JSON (optional row check)")
    ap.add_argument("--reference-features", help="original feature rows NPY (with the above)")
    ap.add_argument("--expect", help="block of configs/eval/expected.json (default: `by_records` of --records)")
    ap.add_argument("--out", help="result JSON (default WORK/eval/never_seen.json; never_seen_png.json for png)")
    a = ap.parse_args()
    if a.reference_rows and a.records != "png":
        ap.error("--reference-rows / --reference-features are the original PNG-union rows: use with --records png")
    exp = expected.load("never_seen", a.expect, a.records)
    t0 = time.time()
    res = NS.run({"FAST": C.load("FAST")}, str(fp.WORK / D007_BIN), str(fp.WORK / D007_PROB), a.workers,
                 records=a.records)
    rows = res.pop("_rows")
    ref = reference_rows(rows, a.reference_rows, a.reference_features) if a.reference_rows else None
    out = {"what": "never-seen class features, D007 and nnHALO", "records": a.records, "results": res,
           "gate": gate(res, exp, ref),
           "not_reproduced": exp["not_reproduced"],
           "seconds": round(time.time() - t0, 1)}
    d = oof_selection.eval_dir()
    d.mkdir(parents=True, exist_ok=True)
    sfx = "" if a.records == REC.DEFAULT else f"_{a.records}"
    np.savez_compressed(d / f"never_seen_rows{sfx}.npz", records=np.array(a.records), cases=np.array(rows["cases"]),
                        n_per_case=rows["n_per_case"], classes=rows["classes"],
                        **{f"F_{k}": v for k, v in rows["F"].items()})
    path = Path(a.out) if a.out else d / f"never_seen{sfx}.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    r = res["results"]["D"]
    print(f"D {res['class_counts']['D']}: contrast AUC {r['contrast']['auc']:.4f} {r['contrast']['ci']}, support "
          f"{r['support']['auc']:.4f} {r['support']['ci']}; drawn by every record {res['headline']['D_all_agree']} of "
          f"{res['headline']['D_multi']} vs M {res['headline']['M_all_agree_frac']:.3f} (area-matched "
          f"{res['headline']['M_all_agree_frac_area_reweighted_to_D']:.4f}, not reproduced); FAST D "
          f"{res['per_model']['FAST']['n_D_bag_mean']:.0f}, "
          f"contrast {res['per_model']['FAST']['contrast_AUC_bag_mean']:.4f}"
          f", support {res['per_model']['FAST']['support_AUC_bag_mean']:.4f}")
    print("gate", "PASS" if out["gate"]["pass"] else f"FAIL {out['gate']['failed']}", f"{out['seconds']} s")
    return 0 if out["gate"]["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
