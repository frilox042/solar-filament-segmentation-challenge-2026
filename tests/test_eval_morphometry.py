"""Object morphology of the morphometry evaluation: arc length, width, branch pixels, on toys.

The end-to-end gate is `scripts/eval_morphometry.py`.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament_seg.evaluation import morphometry as MO  # noqa: E402


def _flat(m):
    ix = np.flatnonzero(m.ravel())
    return ix // m.shape[1] * MO.N + ix % m.shape[1]


def test_geodesic_and_width():
    m = np.zeros((40, 80), bool)
    m[10:15, 10:70] = True                                               # 5 x 60 bar
    a, arc, _, bp, _ = MO.morph(_flat(m))
    assert a == 300 and 50 < arc < 60 and bp == 0
    assert 5 <= MO.width((a, arc)) <= 6
    # negative control: a T-junction has branch pixels
    m[15:30, 38:42] = True
    assert MO.morph(_flat(m))[3] > 0
    # objects below 20 px are not measured
    assert np.isnan(MO.morph(np.arange(10))[1])
