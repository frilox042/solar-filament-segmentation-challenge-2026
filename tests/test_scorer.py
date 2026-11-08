"""The scorer (`filament_seg.scorer`) against an independent implementation of the host notebook's rule.

The host's Self-Evaluation notebook groups ground truth PER ANNOTATOR RECORD and, per record, counts:
  TP = number of (prediction, annotation) PAIRS with IoU > 0.5 (strictly), FP = predictions matching nothing,
  FN = annotations matching nothing;  PQ_w = sum IoU(TP pairs) / (TP + w * FP + w * FN), summed over records.
The reference below computes that directly from masks (no shared code with the scorer) on random toy images with
several records, and is compared with `coco_targets` + `Ref.score` at w = 0.5 and 1.0.
Negative controls: an IoU of exactly 0.5 must NOT count as a hit, and a deliberately wrong (>=) scorer must disagree.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filament_seg.candidates import label_base  # noqa: E402
from filament_seg.scorer import Ref, coco_targets  # noqa: E402

H = W = 2048


def rect(x0, y0, x1, y1):
    m = np.zeros((H, W), bool)
    m[y0:y1, x0:x1] = True
    return m


def poly(x0, y0, x1, y1):
    # pycocotools polygon whose rasterisation covers exactly [x0, x1) x [y0, y1)
    return {"segmentation": [[x0, y0, x1, y0, x1, y1, x0, y1]]}


def poly_mask(a):
    from pycocotools import mask as M
    return M.decode(M.merge(M.frPyObjects(a["segmentation"], H, W))).astype(bool)


def reference_pq(pred_masks, records, w):
    num = den = 0.0
    for anns in records.values():
        if not anns:
            continue
        gts = [poly_mask(a) for a in anns]
        iou = np.array([[(p & g).sum() / max((p | g).sum(), 1) for g in gts] for p in pred_masks]).reshape(
            len(pred_masks), len(gts))
        hit = iou > 0.5
        tp = int(hit.sum())
        fp = int((~hit.any(1)).sum())
        fn = int((~hit.any(0)).sum())
        num += float(iou[hit].sum())
        den += tp + w * fp + w * fn
    return num, den


def toy_image(rng):
    preds, recs = [], {}
    for _ in range(int(rng.integers(2, 6))):
        x, y = int(rng.integers(0, 1800)), int(rng.integers(0, 1800))
        preds.append((x, y, x + int(rng.integers(30, 120)), y + int(rng.integers(30, 120))))
    for r in range(int(rng.integers(1, 4))):
        anns = []
        for (x0, y0, x1, y1) in preds:
            if rng.random() < 0.7:                       # an annotation near the prediction, jittered
                j = int(rng.integers(-25, 25))
                anns.append(poly(max(x0 + j, 0), max(y0 + j, 0), x1 + j, y1 + j))
        if rng.random() < 0.5:                           # an annotation nobody predicted
            anns.append(poly(10, 10, 60, 60))
        recs[f"rec{r}"] = anns
    return preds, recs


def scorer_pq(images):
    v, mp, M, miss, img, k, G = [], [], [], [], [], [], []
    for i, (preds, recs) in enumerate(images):
        p = np.zeros((H, W), bool)
        for b in preds:
            p |= rect(*b)
        L, n, pa, _ = label_base(p)
        a, b_, c, d, _ = coco_targets(L, n, pa, recs)
        v.append(a)
        mp.append(b_)
        M.append(c)
        miss.append(d)
        img.append(np.full(n, i))
        k.append(sum(1 for al in recs.values() if al))
        G.append(sum(len(al) for al in recs.values()))
    Z = {"img": np.concatenate(img).astype(int), "coco_k": np.array(k), "coco_G": np.array(G),
         "coco_v": np.concatenate(v), "coco_mp": np.concatenate(mp), "coco_M": np.concatenate(M),
         "coco_miss": np.concatenate(miss)}
    keep = np.ones(len(Z["img"]), bool)
    return Ref(Z, "coco", keep).score(keep)


def test_scorer_matches_host_rule_on_random_images():
    rng = np.random.default_rng(0)
    images = []
    while len(images) < 6:
        preds, recs = toy_image(rng)
        boxes = [rect(*b) for b in preds]
        if any((a & b).any() for i, a in enumerate(boxes) for b in boxes[i + 1:]):
            continue                                     # predictions are disjoint components, as in the pipeline
        if any((b[2] - b[0]) * (b[3] - b[1]) < 300 for b in preds):
            continue                                     # every prediction is a >= 300 px object
        images.append((preds, recs))
    s = scorer_pq(images)
    assert s["tp"] > 0 and s["fp"] > 0 and s["fn"] > 0      # the toy exercises all three terms
    for w, key in ((0.5, "pq_w05"), (1.0, "pq_w10")):
        num = den = 0.0
        for preds, recs in images:
            a, b = reference_pq([rect(*b) for b in preds], recs, w)
            num += a
            den += b
        assert abs(s[key] - num / den) < 1e-12, (key, s[key], num / den)


def _half_case():
    """A >= 300 px prediction covering exactly half of one annotation's rasterised pixels (IoU == 0.5 exactly)."""
    ann = poly(100, 100, 140, 180)
    g = poly_mask(ann)
    rows = np.flatnonzero(g.any(1))
    assert len(rows) % 2 == 0
    p = g.copy()
    p[rows[len(rows) // 2]:] = False
    iou = (p & g).sum() / (p | g).sum()
    assert iou == 0.5 and p.sum() >= 300
    return p, {"r": [ann]}


def _scorer_on_mask(p, recs):
    L, n, pa, _ = label_base(p)
    v, mp, M, miss, _ = coco_targets(L, n, pa, recs)
    Z = {"img": np.zeros(n, int), "coco_k": np.array([1]), "coco_G": np.array([1]), "coco_v": v, "coco_mp": mp,
         "coco_M": M, "coco_miss": miss}
    keep = np.ones(n, bool)
    return Ref(Z, "coco", keep).score(keep)


def test_iou_exactly_half_is_not_a_hit():
    p, recs = _half_case()
    a, b = reference_pq([p], recs, 0.5)
    assert a == 0.0 and b == 1.0                 # one FP + one FN at w = 0.5, no TP
    s = _scorer_on_mask(p, recs)
    assert s["tp"] == 0 and s["fp"] == 1 and s["fn"] == 1 and s["pq_w05"] == 0.0


def test_negative_control_wrong_threshold_disagrees():
    p, recs = _half_case()
    g = poly_mask(recs["r"][0])
    wrong_tp = int((p & g).sum() / (p | g).sum() >= 0.5)    # what a '>=' scorer would count
    assert wrong_tp == 1 and _scorer_on_mask(p, recs)["tp"] == 0
