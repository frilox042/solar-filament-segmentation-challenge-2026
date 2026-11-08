"""ECE estimator and cross-fitted isotonic of the calibration evaluation, on synthetic scores.

The end-to-end gate is `scripts/eval_calibration.py`.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament_seg.evaluation import calibration as CA  # noqa: E402


def test_ece_controls():
    rng = np.random.default_rng(0)
    n, n_img = 4000, 400
    img = rng.integers(0, n_img, n)
    s = rng.beta(2, 2, n)
    y = (rng.random(n) < s).astype(float)
    assert CA.ece(s, y) < 0.03
    assert CA.ece(np.clip(s + 0.2, 0, 1), y) > 0.05                      # negative control
    q = CA.isotonic_oof(np.clip(s + 0.2, 0, 1), y, img, n_img)
    assert CA.ece(q, y) < 0.03                                            # isotonic repairs the shift
    ties = np.r_[np.zeros(50), np.ones(50)]
    assert CA.ece(ties, ties) == 0.0
