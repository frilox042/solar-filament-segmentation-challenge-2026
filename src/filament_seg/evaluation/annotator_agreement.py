"""Do two annotators agree on WHETHER a filament exists more than on its EXTENT? (296 multi-record observations)

The original computation of the reported values is not part of this repository (only its output values are kept in
`configs/eval/expected.json`). This definition was therefore written from the reported description and the technical
report's method text, and FIXED BEFORE THE FIRST RUN; it is run once and gated, not adjusted until it matches.
Each free choice follows an earlier implementation of "the same statistic, differing only in how two annotators'
objects are linked":

REFERENCE   the competition's per-record annotation PNGs (`records_png`), never the MAGFiLO release. A record's mask is
            the OR of its class files.
            [Note, without changing the definition above: the PNG tree is a PNG export that a reviewer
            cannot obtain. The DEFAULT reference is each record rasterised from the competition COCO
            (`records_coco`, `--records coco`); the PNG reference is kept behind `--records png` only to replicate
            the reported values. The COCO values are regression values produced by this code, not an independent
            measurement.]
POPULATION  the observations with >= 2 annotation records (296 of 707).
OBJECTS     per record: 4-connected components (scipy default) of the record mask with area >= 20 px
            (`filament_instances.separate_instances(min_area=20)`, the instance filter of that date).
PAIR        every ORDERED pair (A, B) of distinct records of the observation, A with >= 1 object:
              existence(A, B) = share of A's objects that overlap (> 0 px) any object of B      (ANY overlap)
              matching(A, B)  = TP / #A objects of the greedy matcher (`eval_harness.match_instances`): each A object
                                in label order takes the free B object of highest IoU; a TP iff that IoU >= 0.5
                                (the artifact's key says ">=", not the host's strict ">")
IMAGE       the mean over the image's ordered pairs; the statistic is the mean over images.
CI          cluster bootstrap over images: 5000 resamples, `default_rng(0).integers(0, n, (5000, n))`, percentile
            2.5 / 97.5 of the resampled mean. The seed and draw of the reported interval are unknown, so only the POINT values are
            gated; the interval is reported beside the reported one.
CONTEXT     matched pairs (pooled over ordered pairs and images) and their median IoU (the row quotes 0.6293 over 4,274
            matched pairs from a later re-run whose unit is not stated); pooled-over-objects difference (+0.2886 in
            the row).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from filament_seg.evaluation import records as REC

MIN_AREA = 20
IOU_THR = 0.5
NB = 5000


def objects(mask: np.ndarray, min_area: int = MIN_AREA) -> tuple[np.ndarray, int]:
    """4-connected components with area >= min_area, relabelled 1..n in label order."""
    lab, n = ndimage.label(mask)
    if n == 0:
        return lab, 0
    area = np.bincount(lab.ravel(), minlength=n + 1)
    keep = np.flatnonzero(area[1:] >= min_area) + 1
    rm = np.zeros(n + 1, np.int32)
    rm[keep] = np.arange(1, len(keep) + 1)
    return rm[lab], len(keep)


def pair_stats(la: np.ndarray, na: int, lb: np.ndarray, nb: int, thr: float = IOU_THR) -> dict:
    """Existence and greedy-matching counts of A's objects against B's (A is the reference side)."""
    if na == 0:
        return {"n": 0, "exist": 0, "tp": 0, "matched_iou": []}
    fg = (la > 0) | (lb > 0)
    a, b = la[fg].astype(np.int64), lb[fg].astype(np.int64)
    cont = np.bincount(a * (nb + 1) + b, minlength=(na + 1) * (nb + 1)).reshape(na + 1, nb + 1)
    inter = cont[1:, 1:]
    exist = int((inter.sum(1) > 0).sum()) if nb else 0
    if nb == 0:
        return {"n": na, "exist": 0, "tp": 0, "matched_iou": []}
    aa = np.bincount(la.ravel(), minlength=na + 1)[1:]
    ab = np.bincount(lb.ravel(), minlength=nb + 1)[1:]
    union = aa[:, None] + ab[None, :] - inter
    iou = np.zeros(inter.shape, float)
    np.divide(inter, union, out=iou, where=union > 0)
    used: set = set()
    tp, mi = 0, []
    for i in range(na):
        if iou[i].max() <= 0:
            continue
        best, bp = -1.0, None
        for k in range(nb):
            if k not in used and iou[i, k] > best:
                best, bp = float(iou[i, k]), k
        if bp is not None and best >= thr:
            tp += 1
            used.add(bp)
            mi.append(best)
    return {"n": na, "exist": exist, "tp": tp, "matched_iou": mi}


def image_stats(recs, min_area: int = MIN_AREA, records: str = REC.DEFAULT) -> dict:
    """Per observation: the ordered-pair rows and the image means (`recs`: the records of the `records` source)."""
    src = REC.source(records)
    labs = [objects(src.record_mask(r), min_area) for r in recs]
    rows = []
    for i, (la, na) in enumerate(labs):
        for j, (lb, nb) in enumerate(labs):
            if i != j and na:
                rows.append(pair_stats(la, na, lb, nb))
    if not rows:
        return {"n_pairs": 0}
    ex = [r["exist"] / r["n"] for r in rows]
    ma = [r["tp"] / r["n"] for r in rows]
    return {"n_pairs": len(rows), "n_records": len(recs), "n_objects": [int(n) for _, n in labs],
            "existence": float(np.mean(ex)), "matching": float(np.mean(ma)),
            "pooled": [sum(r["n"] for r in rows), sum(r["exist"] for r in rows), sum(r["tp"] for r in rows)],
            "matched_iou": [x for r in rows for x in r["matched_iou"]]}


def _one(item):
    case, recs, min_area, records = item
    return case, image_stats(recs, min_area, records)


def boot_mean(v: np.ndarray, seed: int = 0, nb: int = NB) -> list:
    idx = np.random.default_rng(seed).integers(0, len(v), (nb, len(v)))
    return [float(x) for x in np.percentile(v[idx].mean(1), [2.5, 97.5])]


def run(train_case_map, workers: int = 3, root=None, min_area: int = MIN_AREA, records: str = REC.DEFAULT) -> dict:
    """min_area != 20 is only the negative control (a different object definition must miss the published values).
    root: the PNG tree (records "png" only)."""
    src = REC.source(records)
    recs = src.records_by_case(train_case_map, root) if records == "png" else src.records_by_case(train_case_map)
    items = sorted(((c, r, min_area, records) for c, r in recs.items() if len(r) >= 2), key=lambda t: t[0])
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(workers) as ex:
            per = dict(ex.map(_one, items, chunksize=4))
    else:
        per = dict(map(_one, items))
    cases = [c for c, _, _, _ in items if per[c]["n_pairs"]]
    E = np.array([per[c]["existence"] for c in cases])
    M = np.array([per[c]["matching"] for c in cases])
    D = E - M
    pooled = np.array([per[c]["pooled"] for c in cases]).sum(0)
    mi = np.array([x for c in cases for x in per[c]["matched_iou"]])
    k = np.bincount([len(r) for r in recs.values()])
    return {
        "records": records, "min_area": min_area, "n_images": len(cases),
        "n_records_by_k": {str(i): int(k[i]) for i in range(1, len(k)) if k[i]},
        "n_ordered_pairs": int(sum(per[c]["n_pairs"] for c in cases)),
        "existence": {"mean": float(E.mean()), "ci95": boot_mean(E)},
        "matching": {"mean": float(M.mean()), "ci95": boot_mean(M)},
        "existence_minus_matching": {"mean": float(D.mean()), "ci95": boot_mean(D)},
        "context": {"pooled_over_objects": {"n_objects": int(pooled[0]), "existence": float(pooled[1] / pooled[0]),
                                            "matching": float(pooled[2] / pooled[0]),
                                            "difference": float((pooled[1] - pooled[2]) / pooled[0])},
                    "matched_pairs_ordered": int(len(mi)),
                    "median_iou_given_matched": float(np.median(mi)) if len(mi) else None},
        "per_image": {c: {k_: per[c][k_] for k_ in ("existence", "matching", "n_pairs")} for c in cases},
    }
