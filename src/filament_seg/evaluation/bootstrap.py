"""Bootstrap estimators over observations (images), both on the SAME draws `default_rng(seed).integers(0, n, (B, n))`.

* `paired_mean`: the project standard for per-image values -- percentile CI of the MEAN paired difference
  (model minus reference), 5000 resamples.
* `pooled`: for ratio statistics (PQ, pooled mean IoU, m2n fractions) the estimand is a ratio of sums over images,
  not a mean of per-image ratios, so each draw re-sums the counts with the draw's multiplicities. A bagged arm's
  value is the mean over its bags of each bag's pooled ratio. Model and reference share the draws.
"""
from __future__ import annotations

import numpy as np

from filament_seg.evaluation.host_rows import stats

NB = 5000


def weights(n: int, seed: int, nb: int = NB) -> np.ndarray:
    """(nb, n) multiplicity matrix of nb image resamples."""
    idx = np.random.default_rng(seed).integers(0, n, (nb, n))
    return np.stack([np.bincount(r, minlength=n) for r in idx]).astype(float)


def paired_mean(a, b, seed: int, nb: int = NB) -> dict:
    """Mean of the per-image differences a - b and its 95% percentile CI."""
    d = np.asarray(a, float) - np.asarray(b, float)
    idx = np.random.default_rng(seed).integers(0, len(d), (nb, len(d)))
    lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
    return {"mean_diff": float(d.mean()), "ci95": [float(lo), float(hi)], "n": int(len(d)),
            "excludes_zero": bool(lo > 0 or hi < 0)}


def point(bags: list) -> dict:
    """Pooled statistics of per-image counts, mean over bags. bags: list of (n_img, FIELDS) arrays."""
    per = [stats(c.sum(0)) for c in bags]
    return {s: float(np.mean([p[s] for p in per])) for s in per[0]}


def _boot(Wm, bags):
    per = [stats(Wm @ c) for c in bags]
    return {s: np.mean([p[s] for p in per], axis=0) for s in per[0]}


def pooled(Wm: np.ndarray, model_bags: list, ref_bags: list, higher: dict) -> dict:
    """Per statistic: model, reference, delta and the 95% CI of the difference on the shared draws Wm."""
    pm, pr = point(model_bags), point(ref_bags)
    bm, br = _boot(Wm, model_bags), _boot(Wm, ref_bags)
    out = {}
    for s, up in higher.items():
        lo, hi = np.percentile(bm[s] - br[s], [2.5, 97.5])
        out[s] = {"model": pm[s], "reference": pr[s], "delta": pm[s] - pr[s], "ci95": [float(lo), float(hi)],
                  "model_favoured": bool(lo > 0) if up else bool(hi < 0)}
    return out
