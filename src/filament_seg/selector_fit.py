"""Selector fit: the 707 out-of-fold training rows -> the deployed bag of 5 regressors and the keep cut.

Rows: every >= 300 px and every 100-299 px component of the slot's base mask on each image's OUT-OF-FOLD probability
map, with the selector features (`selector_features.image_rows`; member masks: the D036 OOF binary, and P = that same
OOF map >= 0.5) and per-record targets (`scorer.coco_targets`).
Fit: 5 HistGradientBoostingRegressors on image-level 5-way partitions (each leaves one fifth of the images out);
target u = v / k (sum of hit IoU per annotator record); cut = PQ_w05(base components >= 400 px) / 2.
"""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

from filament_seg import selector_tables
from filament_seg.candidates import base_mask
from filament_seg.config import SlotConfig
from filament_seg.maps import load_prob_png
from filament_seg.scorer import Ref, coco_by_stem, coco_targets
from filament_seg.selector_features import image_rows, member_masks, step2_matrix

HGB = dict(max_iter=250, max_leaf_nodes=12, learning_rate=0.05, min_samples_leaf=20,
           l2_regularization=0.0, random_state=0)   # fixed before any held-out number
BAG_SEED = 0


def oof_rows(args: tuple) -> dict:
    """One training image's rows. args = (ci, case, base kind, t, prob dir, members, raw dir, annotations)."""
    from threadpoolctl import threadpool_limits
    ci, case, kind, t, prob_dir, member_specs, raw_dir, anns = args
    with threadpool_limits(1):
        pr = load_prob_png(f"{prob_dir}/{case}.png")
        p = base_mask(kind, pr, t)
        raw = Image.open(f"{raw_dir}/{case}_0000.png")
        members = member_masks(member_specs, pr, lambda m: f"{m.oof_dir}/{case}.png")
        r = image_rows(p, pr, raw, members)
        v, mp, Mh, miss, _ = coco_targets(r["L"], r["n"], r["pa"], anns)
        sv, smp, sM, smiss, _ = coco_targets(r["Ls"], r["ns"], r["sa"], anns)
    k = len(members)
    return dict(ci=ci, big=dict(feats=r["feats"].reshape(-1, 9), mb=r["mb"].reshape(-1, k), cov=r["cov"].reshape(-1, k),
                                ctx=r["ctx"].reshape(-1, 5), v=v, mp=mp, M=Mh, miss=miss),
                small=dict(feats=r["sfeats"].reshape(-1, 9), mb=r["smb"].reshape(-1, k), cov=r["scov"].reshape(-1, k),
                           ctx=r["sctx"].reshape(-1, 5), v=sv, mp=smp, M=sM, miss=smiss))


def floor_table(res: list, Z0: dict) -> tuple[dict, np.ndarray, np.ndarray]:
    """Stack the per-image rows as [all >= 300 px rows by image | all 100-299 px rows by image].
    Returns (Zc: the accounting table `Ref` reads, X: design matrix, area)."""
    res = sorted(res, key=lambda r: r["ci"])
    parts = [(r["ci"], r["big"]) for r in res] + [(r["ci"], r["small"]) for r in res]
    img = np.concatenate([np.full(len(d["v"]), ci) for ci, d in parts]).astype(int)

    def cat(k):
        return np.concatenate([d[k] for _, d in parts])

    Zc = {"img": img, "cases": Z0["cases"], "coco_k": Z0["coco_k"], "coco_G": Z0["coco_G"]}
    for f in ("v", "mp", "M", "miss"):
        Zc[f"coco_{f}"] = cat(f)
    X = step2_matrix(cat("feats"), cat("mb"), cat("cov"), cat("ctx"))
    return Zc, X, cat("feats")[:, 0]


def image_folds(n_img: int, seed: int) -> np.ndarray:
    """Seeded image-level 5-way partition (fold of perm[j] = j % 5)."""
    perm = np.random.default_rng(seed).permutation(n_img)
    f = np.empty(n_img, int)
    f[perm] = np.arange(n_img) % 5
    return f


def fit_bag(X: np.ndarray, y: np.ndarray, img: np.ndarray, base_rows: np.ndarray, seed: int = BAG_SEED) -> list:
    """5 regressors, member b trained on the images outside partition b. The partition is built over the images
    with >= 1 base row; images with only other rows get their own partition (seed + 7)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    ub = np.unique(img[base_rows])
    fpart = np.full(img.max() + 1, -1)
    fpart[ub] = image_folds(len(ub), seed)
    extra = np.setdiff1d(np.unique(img), ub)
    if len(extra):
        fpart[extra] = image_folds(len(extra), seed + 7)
    bag = []
    for b in range(5):
        sel = fpart[img] != b
        bag.append(HistGradientBoostingRegressor(**HGB).fit(X[sel], y[sel]))
    return bag


def build_oof_table(cfg: SlotConfig) -> tuple[dict, np.ndarray, np.ndarray]:
    """All 707 images' rows (cfg.fit_workers processes) -> (Zc, X, area). Refuses the level maps of a staged evaluation
    bundle: they keep only the evaluation's thresholds, not the probabilities the features need."""
    from pathlib import Path

    from filament_seg.evaluation.bundle import MARKER
    if (Path(cfg.oof_prob_dir) / MARKER).exists():
        raise RuntimeError(f"{cfg.oof_prob_dir} holds threshold-level maps from the evaluation bundle ({MARKER}); "
                           "the selector's OOF rows need the real OOF probability maps")
    Z0 = selector_tables.load_from(cfg.selector_tables)
    stem_of, by_stem = coco_by_stem(cfg.coco, cfg.train_case_map)
    cases = [str(c) for c in Z0["cases"]]
    jobs = [(i, c, cfg.base_mask, cfg.core_t, str(cfg.oof_prob_dir), cfg.members,
             str(cfg.raw_train), by_stem[stem_of[c]]) for i, c in enumerate(cases)]
    with ProcessPoolExecutor(cfg.fit_workers) as ex:
        res = list(ex.map(oof_rows, jobs, chunksize=4))
    return floor_table(res, Z0)


def fit(cfg: SlotConfig) -> tuple[list, float, dict]:
    """(bag, cut, info): the deployed selector of the slot."""
    from threadpoolctl import threadpool_limits
    t0 = time.time()
    Zc, X, area = build_oof_table(cfg)
    pm = area >= 400
    R = Ref(Zc, "coco", np.ones(len(area), bool))
    cut = Ref(Zc, "coco", pm).score(np.ones(int(pm.sum()), bool))["pq_w05"] / 2
    with threadpool_limits(cfg.hgb_threads):
        bag = fit_bag(X, R.u, R.img, pm)
    return bag, float(cut), {"oof_rows": int(len(area)), "two_cut": 2 * float(cut), "fit_s": time.time() - t0}
