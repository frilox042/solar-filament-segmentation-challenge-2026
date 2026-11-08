"""The models against a held-out annotator, on the host's metric (paper Table 1 and its Evaluation section).

POPULATION  the 296 OOF training observations with >= 2 annotator records (records = competition COCO image ids with
            >= 1 annotation). UNIT = (observation, held-out record a): 743 units.
REFERENCE   of a unit: every annotation of every record r != a, each record its own host row (FP charged once per
            (prediction, record)). The human and every model are scored against exactly this reference.
ARMS        H_A    the held-out record's raw annotations as prediction rows (the host's own row granularity);
                   H_A400 the same with annotations >= 400 px only (Table 1 footnote d)
            H_B    the 4-connected components of the held-out record's pixel union (what a legal submission could
                   contain: rows may not overlap)
            FAST  the cross-fitted selector keeps (`oof_selection`), 5 bags; D007 = the D007 OOF argmax >= 400 px
AXES        PQ (w 0.5 / 1.0), pooled mean IoU over overlapping pairs, m2n fractions: pooled ratios of sums, CI by
            `bootstrap.pooled`; pixel Dice (torchmetrics include_background=False) per (unit, reference record),
            averaged per observation, CI on the mean paired difference (`bootstrap.paired_mean`).
FAR         observations >= 4 days from every image of their fold's training set (folds: sorted cases[i::5]).
NEGATIVE CONTROL  the human with its own record put back into its reference must score far higher (self-reference).
"""
from __future__ import annotations

import datetime

import numpy as np

from filament_seg.config import SlotConfig
from filament_seg.evaluation import bootstrap as B
from filament_seg.evaluation import host_rows as HR
from filament_seg.evaluation import oof_selection as OS
from filament_seg.maps import load_bin

HIGHER = {"pq_w05": True, "pq_w10": True, "mean_iou": True, "mean_dice": True, "gt_1to0": False,
          "gt_1toge2": False, "pred_0to1": False, "pred_ge2to1": False}
HUMANS = ("H_A0", "H_B0", "H_A400")
MODELS = ("FAST", "D007")
FAR_DAYS = 4.0
D007_MIN_AREA = 400


def _model_rows(lab, n, area, mask, R, RA, RM):
    """Per record: host row counts, and the Dice state (tp, |P|, |T|)."""
    ps = int(np.count_nonzero(mask))
    rows, dice = [], []
    for r in range(len(R)):
        rows.append(HR.row_counts(HR.inter_label(lab, n, R[r]), RA[r], area)[0])
        dice.append([int(np.count_nonzero(mask & RM[r])), ps, int(RM[r].sum())])
    return np.array(rows), np.array(dice)


def image_worker(item: tuple) -> dict:
    """One multi-record observation. item = (i, case, segs per record, {slot: (cfg, rows, keeps, areas)}, d007 path)."""
    from threadpoolctl import threadpool_limits
    i, case, segs, slots, d007_png = item
    with threadpool_limits(1):
        R = [[HR.decode(s) for s in rec] for rec in segs]
        k = len(R)
        RA = [np.array([len(x) for x in r], np.int64) for r in R]
        RM = []
        for r in R:
            m = np.zeros(HR.H * HR.W, bool)
            for x in r:
                m[x] = True
            RM.append(m)
        out = {"i": i, "case": case, "k": k, "models": {}}
        # ---- models: one mask per bag; record identity never enters the mask
        for nm, (cfg, keeps, rows_area) in slots.items():
            bags = []
            for kp, m in zip(keeps, OS.masks_from_keeps(cfg, case, keeps, rows_area), strict=True):
                lab, n, area = HR.components(m)
                # the kept candidates are whole components of the base mask, so relabelling recovers them
                assert np.array_equal(np.sort(area), np.sort(rows_area[kp])), case
                bags.append(_model_rows(lab, n, area, m.ravel(), R, RA, RM))
            out["models"][nm] = bags
        lab, n, area = HR.keep_components(*HR.components(load_bin(d007_png)), D007_MIN_AREA)
        out["models"]["D007"] = [_model_rows(lab, n, area, (lab > 0).ravel(), R, RA, RM)]
        # ---- humans: fair units (a, r != a), and the self-reference negative control (r == a included)
        hum = {h: np.zeros(len(HR.FIELDS)) for h in HUMANS}
        selfref = np.zeros(len(HR.FIELDS))
        for a in range(k):
            for cut, h in ((0, "H_A0"), (400, "H_A400")):
                P = [x for x in R[a] if len(x) >= cut]
                pa = np.array([len(x) for x in P], np.int64)
                for r in range(k):
                    inter = HR.inter_lists(P, R[r]) if P else np.zeros((len(R[r]), 0), np.int64)
                    c = HR.row_counts(inter, RA[r], pa)[0]
                    if r != a:
                        hum[h] += c
                    if cut == 0:
                        selfref += c
            lb, nb, ab = HR.components(RM[a].reshape(HR.H, HR.W))
            for r in range(k):
                if r != a:
                    hum["H_B0"] += HR.row_counts(HR.inter_label(lb, nb, R[r]), RA[r], ab)[0]
        out["humans"], out["selfref"] = hum, selfref
        out["human_dice"] = np.array([[int(np.count_nonzero(RM[a] & RM[r])), int(RM[a].sum()), int(RM[r].sum())]
                                      for a in range(k) for r in range(k) if r != a])
    return out


def fair_counts(rows: np.ndarray) -> np.ndarray:
    """Sum over held-out records a of the model's rows r != a (= (k - 1) x all rows; written out on purpose)."""
    k = len(rows)
    return np.sum([rows[r] for a in range(k) for r in range(k) if r != a], axis=0)


def image_dice(dice_state: np.ndarray) -> float:
    """Mean over held-out a of the mean over r != a of Dice(prediction, record r)."""
    d = HR.dice_fg(dice_state[:, 0], dice_state[:, 1], dice_state[:, 2])
    k = len(d)
    return float(np.mean([np.mean([d[r] for r in range(k) if r != a]) for a in range(k)]))


def far_mask(cases_multi: list, all_cases: list, stem_of: dict) -> tuple[np.ndarray, np.ndarray]:
    """(gap in days to the nearest image of the fold's training set, far = gap >= FAR_DAYS) per multi observation."""
    srt = sorted(all_cases)
    fold = {c: j % 5 for j, c in enumerate(srt)}
    t = {c: datetime.datetime.strptime(stem_of[c][:14], "%Y%m%d%H%M%S") for c in all_cases}
    gap = []
    for c in cases_multi:
        dts = [abs((t[x] - t[c]).total_seconds()) for x in all_cases if fold[x] != fold[c]]
        gap.append(min(dts) / 86400)
    gap = np.array(gap)
    return gap, gap >= FAR_DAYS


def summarise(rows: list, cases_all: list, stem_of: dict, seed: int) -> dict:
    """All comparisons from the per-observation worker outputs."""
    rows = sorted(rows, key=lambda r: r["i"])
    n = len(rows)
    K = np.array([r["k"] for r in rows])
    hum = {h: [np.array([r["humans"][h] for r in rows])] for h in HUMANS}
    mod = {m: [np.array([fair_counts(r["models"][m][b][0]) for r in rows]) for b in range(len(rows[0]["models"][m]))]
           for m in MODELS}
    Wm = B.weights(n, seed)
    out = {"n_images": n, "n_units": int(K.sum()), "k_counts": {"2": int((K == 2).sum()), "3": int((K == 3).sum())},
           "seed": seed, "points": {a: B.point(v) for a, v in {**hum, **mod}.items()}}
    out["negative_control_selfref"] = {"pq_w05_selfref": B.point([np.array([r["selfref"] for r in rows])])["pq_w05"],
                                       "pq_w05_fair": out["points"]["H_A0"]["pq_w05"]}
    out["negative_control_selfref"]["pass"] = bool(out["negative_control_selfref"]["pq_w05_selfref"]
                                                   > out["negative_control_selfref"]["pq_w05_fair"] + 0.2)
    out["pooled"] = {f"{m}_vs_{h}": B.pooled(Wm, mod[m], hum[h], HIGHER) for m in MODELS for h in HUMANS}
    dh = np.array([np.mean(HR.dice_fg(*r["human_dice"].T)) for r in rows])
    dm = {m: np.array([np.mean([image_dice(b[1]) for b in r["models"][m]]) for r in rows]) for m in MODELS}
    out["dice"] = {"H_image_mean": float(dh.mean()), **{f"{m}_image_mean": float(v.mean()) for m, v in dm.items()},
                   **{f"{m}_minus_H": B.paired_mean(v, dh, seed) for m, v in dm.items()}}
    gap, far = far_mask([r["case"] for r in rows], cases_all, stem_of)
    Wf = B.weights(int(far.sum()), seed)
    out["far"] = {"days": FAR_DAYS, "n": int(far.sum()),
                  **{f"{m}_vs_H_A0": {s: v for s, v in B.pooled(Wf, [b[far] for b in mod[m]], [hum["H_A0"][0][far]],
                                                                 HIGHER).items() if s in ("pq_w05", "mean_iou")}
                     for m in ("FAST",)}}
    return out


def run(cfgs: dict[str, SlotConfig], d007_dir: str, workers: int, seeds: tuple[int, ...]) -> dict:
    """Build every multi-record observation (parallel) and summarise it for each bootstrap seed."""
    from concurrent.futures import ProcessPoolExecutor
    any_cfg = next(iter(cfgs.values()))
    recs = HR.records_by_case(any_cfg.coco, any_cfg.train_case_map)
    stem_of = HR.case_stems(any_cfg.train_case_map)
    sels = {nm: OS.build(cfg) for nm, cfg in cfgs.items()}
    cases = sels["FAST"]["cases"]
    assert all(s["cases"] == cases for s in sels.values())
    items = []
    for i, c in enumerate(cases):
        segs = [s for _, s in recs[c]]
        if len(segs) < 2:
            continue
        slots = {}
        for nm, sel in sels.items():
            r = OS.rows_of_image(sel, i)
            slots[nm] = (cfgs[nm], [k[r] for k in sel["keeps"]], sel["area"][r])
        items.append((i, c, segs, slots, f"{d007_dir}/{c}.png"))
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(image_worker, items, chunksize=2))
    return {f"seed_{s}": summarise(rows, cases, stem_of, s) for s in seeds}
