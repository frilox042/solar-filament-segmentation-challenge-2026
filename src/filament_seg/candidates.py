"""Base masks and candidate objects: which connected components the selector scores."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

BIG_MIN = 300      # components >= 300 px: the "big" rows
FLOOR_MIN = 100    # components of 100-299 px: the FLOOR rows


def base_mask(kind: str, pr: np.ndarray, t: float) -> np.ndarray:
    """The base mask from a float32 probability map: "t97" = pr >= t (t = 0.97)."""
    if kind == "t97":
        return pr >= t
    raise ValueError(kind)


def label_base(p: np.ndarray) -> tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    """4-connected components of p relabelled to those >= BIG_MIN px (label order kept).
    Returns (label map L, count n, their areas pa, areas of ALL components pa_all)."""
    pl, pn_all = ndimage.label(p)
    pa_all = np.bincount(pl.ravel(), minlength=pn_all + 1)[1:]
    keep_ids = np.flatnonzero(pa_all >= BIG_MIN) + 1
    remap = np.zeros(pn_all + 1, np.int64)
    remap[keep_ids] = np.arange(1, len(keep_ids) + 1)
    L = remap[pl]
    n = len(keep_ids)
    pa = pa_all[keep_ids - 1]
    return L, n, pa, pa_all


def small_components(p: np.ndarray) -> tuple[np.ndarray, int, np.ndarray, np.ndarray, np.ndarray]:
    """The FLOOR_MIN..BIG_MIN-1 px components.

    Returns (label map Ls, count ns, areas sa, full label map pl, their ids in pl)."""
    pl, pn = ndimage.label(p)
    pa_all = np.bincount(pl.ravel(), minlength=pn + 1)[1:]
    ids = np.flatnonzero((pa_all >= FLOOR_MIN) & (pa_all < BIG_MIN)) + 1
    remap = np.zeros(pn + 1, np.int64)
    remap[ids] = np.arange(1, len(ids) + 1)
    return remap[pl], len(ids), pa_all[ids - 1], pl, ids
