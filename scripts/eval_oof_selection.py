#!/usr/bin/env python3
"""Cross-fitted selector keeps on the 707 OOF training rows (input of every evaluation number), with its gate.

    python scripts/eval_oof_selection.py [--slot nnHALO] [--reference-keeps nnHALO=PKL] [--out JSON]

Writes WORK/eval/<slot>_oof_selection.pkl (rows, 5 bags of keeps and scores, and the identity of its inputs; a cache
whose inputs changed is refused, --rebuild recomputes) and a gate JSON:
  GATE     every bag's pooled per-record PQ (w 0.5, w 1.0) on all 707 == the reported per-bag values
           (configs/eval/expected.json, 1e-12).
  NEG CTL  keeping every candidate (no selector) gives a different PQ, so the gate can fail.
  --reference-keeps  optional: a pickle holding the same keeps ({"keeps": [...]});
           the keeps must be bit-equal. Used once to certify the port; never needed to run the evaluation.
Run with OMP_NUM_THREADS=4 inside a memory scope (peak ~2 GiB with 2-3 workers).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filament_seg import config as C  # noqa: E402
from filament_seg.evaluation import expected, oof_selection  # noqa: E402


def gate(slot: str, sel: dict, ref_keeps: str | None) -> dict:
    exp = expected.load("oof_selection")
    tol = exp["tol"]
    per = oof_selection.pq_per_bag(sel)
    g: dict = {"kept_per_bag": [int(k.sum()) for k in sel["keeps"]], "rows": int(len(sel["area"]))}
    g["pq"] = expected.close([[p["pq_w05"], p["pq_w10"]] for p in per], exp["nnHALO_pq_per_bag_w05_w10"], tol)
    nc = oof_selection.pq_per_bag({**sel, "keeps": [np.ones(len(sel["area"]), bool)]})[0]
    g["negative_control_no_selector"] = {"pq_w05": nc["pq_w05"],
                                         "fails_gate": bool(abs(nc["pq_w05"] - per[0]["pq_w05"]) > tol)}
    if ref_keeps:
        rk = pickle.load(open(ref_keeps, "rb"))["keeps"]
        g["reference_keeps_bit_equal"] = bool(all(np.array_equal(a, b) for a, b in zip(sel["keeps"], rk, strict=True)))
    g["pass"] = bool(g["pq"]["pass"] and g["negative_control_no_selector"]["fails_gate"]
                     and g.get("reference_keeps_bit_equal", True))
    return g


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slot", default="nnHALO", choices=("nnHALO",))
    ap.add_argument("--reference-keeps", action="append", default=[],
                    help="SLOT=PKL (repeatable): reference keeps to compare bit for bit")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--rebuild", action="store_true", help="recompute even if WORK/eval holds a cache")
    ap.add_argument("--out", help="gate JSON (default WORK/eval/oof_selection_gate.json)")
    a = ap.parse_args()
    refs = dict(x.split("=", 1) for x in a.reference_keeps)
    out = {}
    for slot in (a.slot,):
        t0 = time.time()
        sel = oof_selection.build(C.load(slot), threads=a.threads, rebuild=a.rebuild)
        out[slot] = gate(slot, sel, refs.get(slot))
        out[slot]["seconds"] = round(time.time() - t0, 1)
        print(slot, json.dumps(out[slot]), flush=True)
    path = Path(a.out) if a.out else oof_selection.eval_dir() / "oof_selection_gate.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    return 0 if all(v["pass"] for v in out.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
