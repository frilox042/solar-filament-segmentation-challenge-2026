"""Is the per-object selector score a calibrated confidence?

SCORE   s = the cross-fitted selector score `uhat` of each KEPT object (`oof_selection`), clipped to [0, 1]. It is a
        regression of the per-record PQ credit u = v / k, not a probability.
EVENT   y = the object matches (IoU > 0.5) an annotation of at least one record (`coco_M >= 1`).
ECE     10 equal-count bins (np.array_split of the stable argsort of s); sum_b (n_b / N) |mean y_b - mean s_b|.
        Per bag (5 crossfit seeds), then the mean over bags.
ISOTONIC  s -> P(y) (sklearn IsotonicRegression, clipped to [0, 1]) cross-fitted by IMAGE: 5 folds from
        `image_folds(707, seed 0)`; the out-of-fold predictions are scored.
CI      image-cluster bootstrap: per resample (`default_rng(0).integers(0, 707, (5000, 707))`) the objects of the drawn
        images with multiplicity; ECE recomputed with the bins re-formed. The OOF isotonic predictions are resampled,
        not refitted. A bagged CI is the percentile of the per-resample mean over bags.
CONTROLS  on bag 0's real scores, labels drawn from s itself (`default_rng(1)`) must give ECE < 0.03, and the same
        labels against s + 0.2 must give ECE > 0.05: the estimator can tell calibrated from miscalibrated.
TEST SCALE  mean raw score of the kept test objects under the deployed bag vs the pooled OOF kept objects.
"""
from __future__ import annotations

import numpy as np

from filament_seg.selector_fit import image_folds

NBINS = 10
NB = 5000


def ece(s: np.ndarray, y: np.ndarray, nbins: int = NBINS) -> float:
    """Expected calibration error over equal-count bins."""
    if len(s) == 0:
        return float("nan")
    o = np.argsort(s, kind="stable")
    tot = 0.0
    for b in np.array_split(o, nbins):
        if len(b):
            tot += len(b) / len(s) * abs(float(y[b].mean()) - float(s[b].mean()))
    return float(tot)


def ece_boot(s: np.ndarray, y: np.ndarray, img: np.ndarray, boot_idx: np.ndarray) -> np.ndarray:
    """ECE on each image-cluster resample (rows of boot_idx are drawn image indices)."""
    n_img = boot_idx.shape[1]
    order = np.argsort(img, kind="stable")
    starts = np.searchsorted(img[order], np.arange(n_img))
    ends = np.searchsorted(img[order], np.arange(n_img), side="right")
    out = np.empty(len(boot_idx))
    for b, ii in enumerate(boot_idx):
        sel = np.concatenate([order[starts[j]:ends[j]] for j in ii])
        out[b] = ece(s[sel], y[sel])
    return out


def isotonic_oof(s: np.ndarray, y: np.ndarray, img: np.ndarray, n_img: int, seed: int = 0) -> np.ndarray:
    """Out-of-fold isotonic calibration, folds by image."""
    from sklearn.isotonic import IsotonicRegression
    fo = image_folds(n_img, seed)[img]
    q = np.full(len(s), np.nan)
    for f in range(5):
        tr, te = fo != f, fo == f
        ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True).fit(s[tr], y[tr])
        q[te] = ir.predict(s[te])
    return q


def run(sel: dict, nb: int = NB, boot_seed: int = 0) -> dict:
    """ECE (raw and cross-fitted isotonic) of a slot's kept OOF objects, per bag and bag-mean, with CIs and controls."""
    Z = sel["Z"]
    y_all = (np.asarray(Z["coco_M"]) >= 1).astype(float)
    img_all = np.asarray(Z["img"])
    n_img = len(sel["cases"])
    bidx = np.random.default_rng(boot_seed).integers(0, n_img, (nb, n_img))
    per, rb, ib = [], [], []
    for b, kp in enumerate(sel["keeps"]):
        u = sel["uhat"][b][kp]
        s, y, img = np.clip(u, 0, 1), y_all[kp], img_all[kp]
        q = isotonic_oof(s, y, img, n_img, seed=0)
        rb.append(ece_boot(s, y, img, bidx))
        ib.append(ece_boot(q, y, img, bidx))
        per.append({"bag": b, "n_kept": int(kp.sum()), "frac_y": float(y.mean()), "ece_raw": ece(s, y),
                    "ece_isotonic_oof": ece(q, y), "n_clipped": int(((u < 0) | (u > 1)).sum()),
                    "mean_raw_score": float(u.mean())})
    rng = np.random.default_rng(1)
    kp = sel["keeps"][0]
    s0 = np.clip(sel["uhat"][0][kp], 0, 1)
    yc = (rng.random(len(s0)) < s0).astype(float)
    ctl = {"calibrated_synthetic_y_ece": ece(s0, yc), "shifted_s_plus_0.2_ece": ece(np.clip(s0 + 0.2, 0, 1), yc)}
    ctl["pass"] = bool(ctl["calibrated_synthetic_y_ece"] < 0.03 and ctl["shifted_s_plus_0.2_ece"] > 0.05)
    pooled_u = np.concatenate([sel["uhat"][b][k] for b, k in enumerate(sel["keeps"])])
    return {"per_bag": per,
            "ece_raw_bagmean": float(np.mean([p["ece_raw"] for p in per])),
            "ece_raw_ci95": np.percentile(np.mean(rb, 0), [2.5, 97.5]).tolist(),
            "ece_isotonic_oof_bagmean": float(np.mean([p["ece_isotonic_oof"] for p in per])),
            "ece_isotonic_oof_ci95": np.percentile(np.mean(ib, 0), [2.5, 97.5]).tolist(),
            "controls": ctl,
            "oof_kept_pooled": {"n": int(len(pooled_u)), "mean_raw_score": float(pooled_u.mean())}}


def test_scale(cfg, bag: list, cut: float) -> dict:
    """The deployed bag's raw score of the kept TEST objects (no test annotation is read)."""
    from concurrent.futures import ProcessPoolExecutor

    from filament_seg import render
    stems, inv, mf = render.test_setup(cfg)
    with ProcessPoolExecutor(cfg.apply_workers) as ex:
        tr = list(ex.map(render.test_rows, [(cfg, s, inv[s], mf[s]) for s in stems], chunksize=4))
    X, _, _, _, _ = render.design_matrix(tr)
    u = render.scores(bag, X)
    kept = u > cut
    return {"n_kept": int(kept.sum()), "mean_raw_score": float(u[kept].mean()), "cut": float(cut)}
