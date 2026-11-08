"""Classes, stratified AUC and support headline of the never-seen evaluation, on synthetic rows.

The end-to-end gate is `scripts/eval_never_seen.py`.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament_seg.evaluation import never_seen as NS  # noqa: E402


def test_class_precedence():
    assert NS.cls(True, 0.0, True) == "M"
    assert NS.cls(False, 0.05, True) == "D"          # never seen wins over touched
    assert NS.cls(False, 0.3, True) == "A"
    assert NS.cls(False, 0.7, False) == "B"
    assert NS.cls(False, 0.3, False) == "C"


def test_stratified_auc_separates_and_controls():
    rng = np.random.default_rng(0)
    n = 400
    y = np.r_[np.ones(n // 2, bool), np.zeros(n // 2, bool)]
    strata = rng.integers(0, 4, n)
    assert NS.strat_auc(y.astype(float), y, strata) == 1.0
    assert abs(NS.strat_auc(rng.random(n), y, strata) - 0.5) < 0.1          # negative control: no signal
    # a feature that only tracks the stratum carries nothing within strata
    yy = strata >= 2
    assert np.isnan(NS.strat_auc(strata.astype(float), yy, strata))         # no stratum has both classes


def test_headline_counts_all_record_objects():
    F = {"area": np.array([100, 100, 120, 130, 140, 150, 160, 170, 180, 190]),
         "support": np.array([1.0, 0.5, 1.0, 1.0, 0.5, np.nan, 1.0, 1.0, 1.0, 0.5])}
    cl = np.array(["D", "D", "M", "M", "M", "D", "M", "M", "M", "M"])
    k = np.array([2, 2, 2, 3, 3, 1, 2, 3, 3, 2])
    h = NS.headline(F, cl, k)
    assert (h["D_multi"], h["D_all_agree"], h["M_multi"]) == (2, 1, 7)
    assert abs(h["M_all_agree_frac"] - 5 / 7) < 1e-12


def test_rsd():
    assert np.isnan(NS.rsd(np.arange(5.0)))                                   # too few values
    v = np.r_[np.zeros(20), 1e6]
    assert NS.rsd(v) == 0.0                                                    # robust to one outlier
