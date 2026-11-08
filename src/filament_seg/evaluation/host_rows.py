"""Per-record host-notebook accounting on explicit masks, plus the competition annotations it is scored against.

The host Self-Evaluation notebook groups ground truth per annotator RECORD (one COCO image id). Each record is one
"row": every prediction of the observation is matched against that record's annotations, hit = IoU > 0.5 (strict),
TP = number of hit PAIRS, FP = predictions with no hit in the row, FN = annotations with no hit; a row with no
annotation charges every prediction as FP, a row with no prediction every annotation as FN. Its distribution plots
use every OVERLAPPING (annotation, prediction) pair (`plot_distribution`) and the overlap degrees (`plot_m2n_counts`).

`row_counts` returns one row's additive counts (`FIELDS`), so any population's statistic is a ratio of sums.
This is an independent implementation from masks (the selector's `scorer.Ref` works on cached per-object targets);
the evaluation gates check that the two agree.
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy import ndimage

H = W = 2048
FIELDS = ("n_gt", "gt0", "gt1", "gt2", "n_pe", "p0", "p1", "p2", "n_pairs", "s_iou_pairs", "s_dice_pairs", "tp",
          "s_iou_tp", "fp", "fn")
FI = {f: i for i, f in enumerate(FIELDS)}


# ---------------------------------------------------------------------------------------------- annotations
def records_by_case(coco_path, train_case_map) -> dict:
    """case -> list of (record id, [segmentation, ...]) for every record with >= 1 annotation, sorted by record id.
    Records are the competition COCO image ids `<record>-<image stem>`."""
    full = json.load(open(train_case_map))
    full = full.get("train", full)
    stem_of = {k: os.path.splitext(v if isinstance(v, str) else v["image"])[0].lower() for k, v in full.items()}
    coco = json.load(open(coco_path))
    rec_ids: dict = {}
    for im in coco["images"]:
        rec_ids.setdefault(im["id"].split("-", 1)[1].lower(), []).append(im["id"])
    anns: dict = {}
    for an in coco["annotations"]:
        anns.setdefault(an["image_id"], []).append(an["segmentation"])
    return {case: [(rid, anns[rid]) for rid in sorted(rec_ids[stem]) if rid in anns]
            for case, stem in stem_of.items() if stem in rec_ids}


def case_stems(train_case_map) -> dict:
    """case -> competition image stem (lower case)."""
    full = json.load(open(train_case_map))
    full = full.get("train", full)
    return {k: os.path.splitext(v if isinstance(v, str) else v["image"])[0].lower() for k, v in full.items()}


def decode(seg) -> np.ndarray:
    """Sorted flat pixel indices of one COCO polygon annotation (pycocotools rasterisation, 2048 x 2048)."""
    from pycocotools import mask as M
    return np.flatnonzero(M.decode(M.merge(M.frPyObjects(seg, H, W)))).astype(np.int64)


# ---------------------------------------------------------------------------------------------- intersections
def components(mask: np.ndarray) -> tuple[np.ndarray, int, np.ndarray]:
    """4-connected components (scipy's default cross): (labels, count, areas)."""
    lab, n = ndimage.label(mask)
    return lab, n, np.bincount(lab.ravel(), minlength=n + 1)[1:].astype(np.int64)


def keep_components(lab: np.ndarray, n: int, area: np.ndarray, min_area: int) -> tuple[np.ndarray, int, np.ndarray]:
    """Relabel to the components with area >= min_area (label order kept)."""
    kk = np.flatnonzero(area >= min_area) + 1
    rm = np.zeros(n + 1, np.int32)
    rm[kk] = np.arange(1, len(kk) + 1)
    return rm[lab], len(kk), area[kk - 1]


def inter_label(lab: np.ndarray, n: int, G: list) -> np.ndarray:
    """(n_gt, n) intersections between a label image (1..n, non-overlapping) and annotation index arrays."""
    flat = lab.ravel()
    out = np.zeros((len(G), n), np.int64)
    if n:
        for i, g in enumerate(G):
            out[i] = np.bincount(flat[g], minlength=n + 1)[1:]
    return out


def _bbox(ix):
    r, c = ix // W, ix % W
    return r.min(), r.max(), c.min(), c.max()


def inter_lists(P: list, G: list) -> np.ndarray:
    """(n_gt, n_pred) intersections between two lists of sorted index arrays (either list may self-overlap,
    as raw annotations of one record can)."""
    out = np.zeros((len(G), len(P)), np.int64)
    bp = [_bbox(p) for p in P]
    for i, g in enumerate(G):
        bg = _bbox(g)
        for j, p in enumerate(P):
            b = bp[j]
            if b[0] > bg[1] or bg[0] > b[1] or b[2] > bg[3] or bg[2] > b[3]:
                continue
            out[i, j] = len(np.intersect1d(g, p, assume_unique=True))
    return out


# ---------------------------------------------------------------------------------------------- one row
def row_counts(inter: np.ndarray, ga, pa) -> tuple[np.ndarray, np.ndarray]:
    """One host row. inter: (n_gt, n_pred) intersections; ga, pa: areas. Returns (counts over FIELDS, the IoUs of the
    overlapping pairs)."""
    ng, npd = inter.shape
    c = np.zeros(len(FIELDS))
    c[FI["n_gt"]], c[FI["n_pe"]] = ng, npd
    if ng == 0:                                   # host branch: every prediction is an FP
        c[FI["fp"]] = c[FI["p0"]] = npd
        return c, np.zeros(0)
    if npd == 0:                                  # host branch: every annotation is an FN
        c[FI["fn"]] = c[FI["gt0"]] = ng
        return c, np.zeros(0)
    ga = np.asarray(ga, float)[:, None]
    pa = np.asarray(pa, float)[None, :]
    U = ga + pa - inter
    iou = np.where(U > 0, inter / np.maximum(U, 1), 0.0)
    dice = 2 * inter / (ga + pa)
    hit, ov = iou > 0.5, inter > 0
    gd, pd = ov.sum(1), ov.sum(0)
    c[FI["gt0"]], c[FI["gt1"]], c[FI["gt2"]] = (gd == 0).sum(), (gd == 1).sum(), (gd >= 2).sum()
    c[FI["p0"]], c[FI["p1"]], c[FI["p2"]] = (pd == 0).sum(), (pd == 1).sum(), (pd >= 2).sum()
    c[FI["n_pairs"]], c[FI["s_iou_pairs"]], c[FI["s_dice_pairs"]] = ov.sum(), iou[ov].sum(), dice[ov].sum()
    c[FI["tp"]], c[FI["s_iou_tp"]] = hit.sum(), iou[hit].sum()
    c[FI["fp"]], c[FI["fn"]] = (hit.sum(0) == 0).sum(), (hit.sum(1) == 0).sum()
    return c, iou[ov]


def stats(S: np.ndarray) -> dict:
    """Statistics of summed counts S (..., FIELDS): PQ (w 0.5, 1.0), pooled pair means, m2n fractions."""
    def f(k):
        return S[..., FI[k]]
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"pq_w05": f("s_iou_tp") / (f("tp") + 0.5 * f("fp") + 0.5 * f("fn")),
                "pq_w10": f("s_iou_tp") / (f("tp") + f("fp") + f("fn")),
                "mean_iou": f("s_iou_pairs") / f("n_pairs"), "mean_dice": f("s_dice_pairs") / f("n_pairs"),
                "gt_1to0": f("gt0") / f("n_gt"), "gt_1toge2": f("gt2") / f("n_gt"),
                "pred_0to1": f("p0") / f("n_pe"), "pred_ge2to1": f("p2") / f("n_pe")}


# ---------------------------------------------------------------------------------------------- pixel Dice
def dice_fg(tp, ps, ts) -> np.ndarray:
    """Foreground Dice 2|P&T| / (|P| + |T|) per sample: torchmetrics `DiceScore(num_classes=2,
    include_background=False, average="micro", input_format="index")` of one (prediction, target) pair
    (checked against the literal metric in tests/test_eval_dice.py). 0/0 (both empty) is NaN here; it cannot occur
    in this evaluation because every reference record has at least one annotation."""
    tp, ps, ts = (np.asarray(x, float) for x in (tp, ps, ts))
    with np.errstate(invalid="ignore", divide="ignore"):
        return 2 * tp / (ps + ts)
