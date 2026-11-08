#!/usr/bin/env python3
"""Annotators agree on whether a filament EXISTS far more than on its EXTENT (296 multi-record observations).

    python scripts/eval_annotator_agreement.py [--records coco|png] [--workers 3] [--out JSON]

Needs the competition training COCO and the case mapping (`--records coco`, the default), or, for `--records png`
only, the historical per-record annotation PNGs under FILAMENT_EXTRACTED/training. No model output is read.
--records png replicates the reported values (block `by_records.png` of configs/eval/expected.json); --records coco
gates against this code's own COCO run (block `by_records.coco`: regression values produced by this code, not an
independent measurement).
Method (fixed before the first run, because the original computation is not part of this repository):
`filament_seg.evaluation.annotator_agreement`.
GATE (png): the three reported point values (existence 0.8393, matching 0.5629,
difference +0.2765) to 1e-12, n = 296; the reported context values (4,274 matched pairs, median IoU 0.6293,
pooled difference +0.2886) to their printed precision. The bootstrap interval is reported, not gated (unknown seed).
NEGATIVE CONTROL: the same statistic on objects without the 20 px filter must miss the reported existence by > 1e-3.
CPU only; < 1 GiB; about 3 min per run with 3 workers (the control is a second run).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filament_seg import config as C  # noqa: E402
from filament_seg.evaluation import annotator_agreement as AA  # noqa: E402
from filament_seg.evaluation import expected, oof_selection  # noqa: E402
from filament_seg.evaluation import records as REC  # noqa: E402


def gate(res: dict, nc: dict, exp: dict) -> dict:
    t, tp = exp["tol"], exp["tol_printed"]
    pr, cx = exp["printed"], res["context"]
    ch = {
        "n_images": {"mine": res["n_images"], "published": exp["n_images"],
                     "pass": res["n_images"] == exp["n_images"]},
        "points": expected.close([res["existence"]["mean"], res["matching"]["mean"],
                                  res["existence_minus_matching"]["mean"]],
                                 [exp["existence_mean"], exp["matching_mean"],
                                  exp["existence_minus_matching_mean"]], t),
        "printed.matched_pairs": {"mine": cx["matched_pairs_ordered"], "published": pr["matched_pairs"],
                                  "pass": cx["matched_pairs_ordered"] == pr["matched_pairs"]},
        "printed.median_iou_given_matched": expected.close(cx["median_iou_given_matched"],
                                                           pr["median_iou_given_matched"], tp),
        "printed.pooled_over_objects_difference": expected.close(cx["pooled_over_objects"]["difference"],
                                                                 pr["pooled_over_objects_difference"], tp),
    }
    d = abs(nc["existence"]["mean"] - exp["existence_mean"])
    ch["negative_control_no_area_filter"] = {"min_area": nc["min_area"], "existence": nc["existence"]["mean"],
                                             "abs_diff": d, "pass": bool(d > 1e-3)}
    return {"pass": all(v["pass"] for v in ch.values()), "failed": [k for k, v in ch.items() if not v["pass"]],
            "checks": ch}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=3)
    REC.add_argument(ap)
    ap.add_argument("--expect", help="block of configs/eval/expected.json (default: `by_records` of --records)")
    ap.add_argument("--out", help="result JSON (default WORK/eval/annotator_agreement.json)")
    a = ap.parse_args()
    exp = expected.load("annotator_agreement", a.expect, a.records)
    t0 = time.time()
    cmap = C.train_case_map()
    res = AA.run(cmap, a.workers, records=a.records)
    nc = AA.run(cmap, a.workers, min_area=1, records=a.records)
    nc.pop("per_image")
    out = {"what": "annotator existence vs extent agreement", "records": a.records, "results": res,
           "negative_control_run": nc,
           "published_ci95_not_gated": exp["published_ci95_not_gated"], "gate": gate(res, nc, exp),
           "seconds": round(time.time() - t0, 1)}
    sfx = "" if a.records == REC.DEFAULT else f"_{a.records}"
    path = Path(a.out) if a.out else oof_selection.eval_dir() / f"annotator_agreement{sfx}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, default=float))
    print(f"existence {res['existence']['mean']:.4f} {res['existence']['ci95']}  "
          f"matching {res['matching']['mean']:.4f} {res['matching']['ci95']}  "
          f"difference {res['existence_minus_matching']['mean']:+.4f} {res['existence_minus_matching']['ci95']}  "
          f"(n {res['n_images']}, records {a.records})")
    print("gate", "PASS" if out["gate"]["pass"] else f"FAIL {out['gate']['failed']}", f"{out['seconds']} s")
    return 0 if out["gate"]["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
