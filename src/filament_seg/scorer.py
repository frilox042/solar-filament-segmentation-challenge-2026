"""The scorer: per-record Panoptic Quality exactly as the host's Self-Evaluation notebook counts it.

Ground truth is grouped PER ANNOTATOR RECORD (COCO image id). For each object o and record r, m_r = number of
annotations of r it overlaps at IoU > 0.5 (strict). Over an image set:

    w = 0.5:  PQ = sum IoU(hits) / (0.5 * G + sum_o 0.5 * sum_r max(m_r, 1))    (the notebook counts PAIRS)
    w = 1.0:  PQ = sum IoU(hits) / (G + sum_o #records the object misses)

G = number of annotations. These targets are also the selector's regression target u = v / k (k = records).
"""
from __future__ import annotations

import json
import os

import numpy as np


def per_record_hits(iou: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per object: sum of IoU over hits (> 0.5) and number of hit pairs, against one record's annotations."""
    hit = iou > 0.5
    return (iou * hit).sum(1), hit.sum(1)


def coco_targets(L: np.ndarray, n: int, pa: np.ndarray, anns: dict) -> tuple:
    """Per-record targets of n objects against one image's COCO annotations grouped by annotator record.

    Returns per object: v (sum of IoU over hits), mp (hit pairs), M (records hit), miss (records missed); plus the
    hit list (record, annotation, object). A hit is IoU > 0.5 (strict, as the host notebook)."""
    from pycocotools import mask as M
    v, mp, Mh, miss = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    hits = []   # (record index, annotation index, object local index)
    flat = L.ravel()
    for ri, al in enumerate(anns.values()):
        if not al:
            continue
        ious = np.zeros((n, len(al)))
        for j, an in enumerate(al):
            g = M.decode(M.merge(M.frPyObjects(an["segmentation"], 2048, 2048))).astype(bool)
            aa = int(g.sum())
            if n and aa:
                J = np.bincount(flat[g.ravel()], minlength=n + 1)[1:]
                ious[:, j] = np.where(J > 0, J / np.maximum(pa + aa - J, 1), 0.0)
        if n:
            s, m_ = per_record_hits(ious)
            v += s
            mp += m_
            Mh += m_ > 0
            miss += m_ == 0
            for o, j in zip(*np.nonzero(ious > 0.5), strict=True):
                hits.append((ri, int(j), int(o)))
    return v, mp, Mh, miss, hits


class Ref:
    """Exact per-record host accounting on one reference, additive over objects.
    w = 0.5: PQ = SumIoU / (0.5 G + sum_o 0.5 * sum_r max(m_r, 1))   [host counts pairs]
    w = 1.0: PQ = SumIoU / (G + sum_o #records missed)"""

    def __init__(self, Z, ref, obj_mask):
        self.ref, self.m = ref, obj_mask
        self.img = Z["img"][obj_mask]
        self.k = Z[f"{ref}_k"][self.img].astype(float)
        self.v = Z[f"{ref}_v"][obj_mask]
        self.mp = Z[f"{ref}_mp"][obj_mask]
        self.M = Z[f"{ref}_M"][obj_mask]
        self.miss = Z[f"{ref}_miss"][obj_mask]
        self.G_img = Z[f"{ref}_G"].astype(float)
        self.k_img = Z[f"{ref}_k"].astype(float)
        self.n_img = len(self.G_img)
        self.den05_o = 0.5 * (self.mp + self.miss)   # = 0.5 * sum_r max(m_r, 1) since m_r>=1 counted in mp
        self.den10_o = self.miss
        self.u = self.v / np.maximum(self.k, 1)
        self.h = self.M / np.maximum(self.k, 1)

    def per_image(self, keep):
        n = self.n_img
        num = np.bincount(self.img[keep], weights=self.v[keep], minlength=n)
        d05 = 0.5 * self.G_img + np.bincount(self.img[keep], weights=self.den05_o[keep], minlength=n)
        d10 = self.G_img + np.bincount(self.img[keep], weights=self.den10_o[keep], minlength=n)
        return num, d05, d10

    def score(self, keep, imgs=None):
        num, d05, d10 = self.per_image(keep)
        sel = np.ones(self.n_img, bool) if imgs is None else imgs
        objsel = keep & sel[self.img]
        tp = int(self.mp[objsel].sum())
        fp = int(self.miss[objsel].sum())
        G = int(self.G_img[sel].sum())
        return {"pq_w05": float(num[sel].sum() / d05[sel].sum()), "pq_w10": float(num[sel].sum() / d10[sel].sum()),
                "tp": tp, "fp": fp, "fn": G - tp, "sum_iou": float(num[sel].sum()), "kept": int(objsel.sum())}


def coco_by_stem(coco_path, train_case_map) -> tuple[dict, dict]:
    """(stem_of: case -> image stem, by_stem: stem -> {record id: [annotations]}) from the competition COCO JSON."""
    full = json.load(open(train_case_map))
    full = full.get("train", full)
    stem_of = {k: os.path.splitext(v if isinstance(v, str) else v["image"])[0].lower() for k, v in full.items()}
    coco = json.load(open(coco_path))
    by_stem: dict = {}
    for im in coco["images"]:
        by_stem.setdefault(im["id"].split("-", 1)[1].lower(), {})[im["id"]] = []
    for an in coco["annotations"]:
        by_stem[an["image_id"].split("-", 1)[1].lower()][an["image_id"]].append({"segmentation": an["segmentation"]})
    return stem_of, by_stem
