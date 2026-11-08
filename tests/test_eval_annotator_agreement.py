"""Pair statistics of the annotator-agreement evaluation, on synthetic records.

The end-to-end gate is `scripts/eval_annotator_agreement.py`.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament_seg.evaluation import annotator_agreement as AA  # noqa: E402


def _rect(m, y0, y1, x0, x1):
    m[y0:y1, x0:x1] = True
    return m


def test_pair_stats_existence_vs_matching():
    a = np.zeros((64, 64), bool)
    _rect(a, 2, 12, 2, 12)        # object 1: drawn the same by B
    _rect(a, 20, 30, 2, 12)       # object 2: B draws a much larger shape over it (exists, does not match)
    _rect(a, 40, 50, 40, 50)      # object 3: B has nothing there
    b = np.zeros((64, 64), bool)
    _rect(b, 2, 12, 2, 12)
    _rect(b, 18, 40, 0, 30)
    la, na = AA.objects(a)
    lb, nb = AA.objects(b)
    r = AA.pair_stats(la, na, lb, nb)
    assert (r["n"], r["exist"], r["tp"]) == (3, 2, 1)
    # self pair: everything exists and matches at IoU 1
    s = AA.pair_stats(la, na, la, na)
    assert (s["exist"], s["tp"]) == (3, 3) and s["matched_iou"] == [1.0, 1.0, 1.0]
    # negative control: nothing in B -> existence and matching 0
    z = AA.pair_stats(la, na, np.zeros_like(la), 0)
    assert (z["exist"], z["tp"]) == (0, 0)


def test_iou_threshold_is_inclusive_and_area_filter():
    a = _rect(np.zeros((40, 40), bool), 0, 10, 0, 10)          # 100 px
    b = _rect(np.zeros((40, 40), bool), 0, 10, 0, 5)           # 50 px inside: IoU exactly 0.5
    la, na = AA.objects(a)
    lb, nb = AA.objects(b)
    assert AA.pair_stats(la, na, lb, nb)["tp"] == 1             # ">= 0.5", as the artifact's key says
    c = _rect(np.zeros((40, 40), bool), 30, 34, 30, 34)        # 16 px: dropped by the 20 px filter
    assert AA.objects(c)[1] == 0 and AA.objects(c, min_area=1)[1] == 1
