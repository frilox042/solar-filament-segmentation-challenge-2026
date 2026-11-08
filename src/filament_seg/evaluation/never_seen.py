"""The never-seen class: what distinguishes the annotated filaments a model never fires on (707 OOF).

Port of the original per-object rows, features, classes and area-stratified AUCs, and of the per-bag
reclassification for nnHALO. Same arithmetic; the full-image dilations are done on crops that contain their whole reach, which selects the same pixels.

REFERENCE  the union of every record's mask, from the per-record source `records` (`records.source`): `coco` (DEFAULT,
           each record rasterised from the competition COCO; the union = Dataset005 labelsTr) or `png` (the historical
           `extracted/` PNG tree, = the union labels on which the reported values were measured; kept only
           to replicate them). A GT object = a 4-connected component of that union (`ndimage.label` order). Support
           uses each record's mask from the same source. COCO values are regression values produced by this code, not an independent measurement.
ROWS (D007)  pred = the D007 OOF argmax binary; matched = some component >= 400 px of it has IoU > 0.5 with the
           object; touched_by_deleted = the object overlaps a component < 400 px; p_max = max D007 OOF probability on
           the object; contrast = mean raw intensity on the object minus the median over the ring
           dilate_12(o) & ~dilate_3(o) (raw = channel 0 of the nnU-Net input, uint16 DN).
FEATURES  on the object's bounding box padded by 32 px: hp = raw - gaussian_filter(raw, 8);
           clutter = robust SD (1.4826 MAD) of hp on dilate_30(o) & ~dilate_4(o) & ~union; interior_texture = robust
           SD of hp on the object (objects > 20 px) / clutter; width = area / skeleton pixels; rR = distance of the
           object centroid from the fitted disk centre / fitted radius (per-frame limb fit, `analysis/datastudy_
           diskfit.json`, image-derived); support (observations with >= 2 records) = share of records whose mask covers
           >= 20% of the object.
CLASSES    M matched; D never seen (p_max < 0.1); A touched by a deleted component; B p_max > 0.5; else C (not
           analysed). Precedence in that order.
AUC        class vs M within 8 log-area strata (edges: octiles of the pooled class + M log areas), pair-weighted,
           folded to >= 0.5; CI = 500 image resamples from ONE `default_rng(0)` stream consumed in the order
           D, A, B x contrast, clutter, interior_texture, width, rR, support (the original count, below the usual 5000:
           kept, because the interval would move).
HEADLINE   on multi-record observations: D and M objects drawn by EVERY record
           (support == 1), overall and by k; M's all-record share reweighted to D's distribution over the same 8
           log-area strata (sum over strata of D's stratum share x M's within-stratum rate).
PER SLOT   for each bag of FAST: matched = a KEPT object has IoU > 0.5; p_max on the slot's own OOF map;
           touched_by_deleted = overlaps the slot's base mask outside the kept objects; same features, same AUC block
           with a fresh `default_rng(0)` per bag; headline = mean over the 5 bags of the per-bag D-vs-M AUC.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.stats import rankdata
from skimage.morphology import skeletonize

from filament_seg import filament_paths as fp
from filament_seg.candidates import base_mask
from filament_seg.evaluation import oof_selection as OS
from filament_seg.evaluation import records as REC
from filament_seg.maps import load_bin, load_prob_png

N = 2048
MIN_AREA, IOU_T = 400, 0.5
FEATS = ["contrast", "clutter", "interior_texture", "width", "rR", "support"]
NBOOT = 500
NBAG = 5
DISKFIT = "analysis/datastudy_diskfit.json"     # relative to WORK (configs/eval/inputs.json)


def _crop(sl, pad):
    return (slice(max(0, sl[0].start - pad), min(N, sl[0].stop + pad)),
            slice(max(0, sl[1].start - pad), min(N, sl[1].stop + pad)))


def rsd(v):
    v = v[np.isfinite(v)]
    return float(1.4826 * np.median(np.abs(v - np.median(v)))) if v.size > 10 else np.nan


def cls(matched: bool, p_max: float, touched: bool) -> str:
    if matched:
        return "M"
    if p_max < 0.1:
        return "D"
    if touched:
        return "A"
    if p_max > 0.5:
        return "B"
    return "C"


# ---------------------------------------------------------------------------------------------- per image
def image_rows(item) -> dict:
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):
        return _image(*item)


def _image(case, recs_src, records, raw_png, d007_bin, d007_prob, fit, slots):
    src = REC.source(records)
    rec_masks = [src.record_mask(r) for r in recs_src]
    gt = np.zeros((N, N), bool)
    for m_ in rec_masks:
        gt |= m_
    gl, gn = ndimage.label(gt)
    out = {"case": case, "n_records": len(recs_src), "n": gn}
    if gn == 0:
        return out
    pr = load_bin(d007_bin)
    prob = load_prob_png(d007_prob)
    img = np.asarray(Image.open(raw_png)).astype(np.float32)
    img = img[..., 0] if img.ndim == 3 else img
    hp = img - ndimage.gaussian_filter(img, 8)
    recs = rec_masks if len(rec_masks) > 1 else []
    pl, pn = ndimage.label(pr)
    pa = np.bincount(pl.ravel(), minlength=pn + 1)[1:] if pn else np.zeros(0, np.int64)
    surv = np.zeros(pn + 1, bool)
    if pn:
        surv[1:] = pa >= MIN_AREA
    ga = np.bincount(gl.ravel(), minlength=gn + 1)[1:]
    objs = ndimage.find_objects(gl)
    # slot arms: kept label image + areas, own map, deleted mask (base mask outside the kept objects)
    arms = {}
    for nm, (cfg, keeps, rows_area) in slots.items():
        cl, area = OS.candidate_labels(cfg, case)
        assert np.array_equal(area, rows_area), f"candidate areas != OOF rows ({case})"
        spr = load_prob_png(f"{cfg.oof_prob_dir}/{case}.png")
        full = base_mask(cfg.base_mask, spr, cfg.core_t)
        for b, k in enumerate(keeps):
            ids = np.flatnonzero(k)
            rm = np.zeros(len(area) + 1, np.int64)
            rm[ids + 1] = np.arange(1, len(ids) + 1)
            lab = rm[cl]
            arms[f"{nm}_b{b}"] = (lab, len(ids), area[ids].astype(float), spr, full & (lab == 0))
    rows = {k: [] for k in ("area", "iou", "matched", "touched_by_deleted", "p_max", "contrast", "clutter",
                            "interior_texture", "width", "rR", "support", "support50")}
    arm_rows = {a: {"iou": [], "p_max": [], "touched_by_deleted": []} for a in arms}
    for g in range(1, gn + 1):
        sl = objs[g - 1]
        # ---- per-object rows (reach of dilate_12 is 12 px: a 13 px margin contains it)
        c1 = _crop(sl, 13)
        m1 = gl[c1] == g
        lab = pl[c1][m1] if pn else np.zeros(0, int)
        best = 0.0
        for c in np.unique(lab[lab > 0]):
            if not surv[c]:
                continue
            it = int((lab == c).sum())
            best = max(best, it / (int(ga[g - 1]) + int(pa[c - 1]) - it))
        touched = bool(pn and np.any(~surv[lab[lab > 0]])) if pn else False
        ring = ndimage.binary_dilation(m1, iterations=12) & ~ndimage.binary_dilation(m1, iterations=3)
        im1 = img[c1]
        contrast = float(im1[m1].mean() - np.median(im1[ring])) if ring.any() else np.nan
        p_max = float(prob[c1][m1].max())
        # ---- features on the 32 px padded box
        c2 = _crop(sl, 32)
        m = gl[c2] == g
        assert int(m.sum()) == int(ga[g - 1])
        ring2 = ndimage.binary_dilation(m, iterations=30) & ~ndimage.binary_dilation(m, iterations=4) & ~gt[c2]
        h = hp[c2]
        clut = rsd(h[ring2])
        inner = rsd(h[m]) if m.sum() > 20 else np.nan
        sk = skeletonize(m).sum()
        ys, xs = np.nonzero(m)
        rr = np.hypot(ys.mean() + c2[0].start - fit["fit_y"], xs.mean() + c2[1].start - fit["fit_x"]) / fit["fit_R"]
        sup = sup50 = np.nan
        if recs:
            cov = [(rm[c2] & m).sum() for rm in recs]
            sup = float(np.mean([c >= 0.2 * m.sum() for c in cov]))
            sup50 = float(np.mean([c >= 0.5 * m.sum() for c in cov]))     # negative control only
        for k, v in (("area", int(ga[g - 1])), ("iou", float(best)), ("matched", bool(best > IOU_T)),
                     ("touched_by_deleted", touched), ("p_max", p_max), ("contrast", contrast), ("clutter", clut),
                     ("interior_texture", inner / clut if clut and np.isfinite(inner) else np.nan),
                     ("width", m.sum() / max(sk, 1)), ("rR", float(rr)), ("support", sup), ("support50", sup50)):
            rows[k].append(v)
        # ---- slot arms (claims_on_shipped C2)
        for a, (alab, an, aarea, apr, adel) in arms.items():
            lb = alab[c1][m1]
            iou = 0.0
            if an:
                J = np.bincount(lb, minlength=an + 1)[1:]
                nz = J > 0
                iou = float((J[nz] / (ga[g - 1] + aarea[nz] - J[nz])).max()) if nz.any() else 0.0
            arm_rows[a]["iou"].append(iou)
            arm_rows[a]["p_max"].append(float(apr[c1][m1].max()))
            arm_rows[a]["touched_by_deleted"].append(bool(adel[c1][m1].any()))
    out["rows"] = {k: np.asarray(v) for k, v in rows.items()}
    out["arms"] = {a: {k: np.asarray(v) for k, v in r.items()} for a, r in arm_rows.items()}
    return out


# ---------------------------------------------------------------------------------------------- statistics
def strat_auc(x, y, strata):
    num = den = 0.0
    for s in np.unique(strata):
        m = strata == s
        n1 = y[m].sum()
        n0 = m.sum() - n1
        if n1 and n0:
            rk = rankdata(x[m])
            a = (rk[y[m]].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
            num += a * n1 * n0
            den += n1 * n0
    return num / den if den else np.nan


def auc_block(F: dict, classes: np.ndarray, img_id: np.ndarray, classes_to=("D", "A", "B")) -> dict:
    """The AUC block: one default_rng(0) stream, consumed class by class and feature by feature."""
    res, rng = {}, np.random.default_rng(0)
    for c in classes_to:
        sel = np.flatnonzero((classes == c) | (classes == "M"))
        y = classes[sel] == c
        la = np.log(F["area"][sel].astype(float))
        strata = np.digitize(la, np.quantile(la, np.linspace(0, 1, 9)[1:-1]))
        im = img_id[sel]
        res[c] = {"n": int(y.sum()), "n_M": int((~y).sum())}
        for ft in FEATS:
            x = F[ft][sel].astype(float)
            ok = np.isfinite(x)
            if ok.sum() < 50 or y[ok].sum() < 10:
                continue
            a = strat_auc(x[ok], y[ok], strata[ok])
            flip = a < 0.5
            a = max(a, 1 - a)
            imgs = np.unique(im[ok])
            bs = []
            for _ in range(NBOOT):
                pick = rng.choice(imgs, len(imgs))
                idx = np.concatenate([np.flatnonzero((im == i) & ok) for i in pick])
                b = strat_auc(x[idx], y[idx], strata[idx])
                bs.append(1 - b if flip else b)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            res[c][ft] = {"auc": float(a), "ci": [float(lo), float(hi)],
                          "median_class": float(np.nanmedian(x[ok & y])), "median_M": float(np.nanmedian(x[ok & ~y])),
                          "n_used": int(ok.sum())}
    return res


def _dm(F: dict, classes: np.ndarray, ft: str):
    """(x, y, strata) of the class-D vs M comparison of one feature (finite values), 8 log-area strata."""
    sel = np.flatnonzero((classes == "D") | (classes == "M"))
    y = classes[sel] == "D"
    la = np.log(F["area"][sel].astype(float))
    strata = np.digitize(la, np.quantile(la, np.linspace(0, 1, 9)[1:-1]))
    x = F[ft][sel].astype(float)
    ok = np.isfinite(x)
    return x[ok], y[ok], strata[ok]


def headline(F: dict, classes: np.ndarray, k: np.ndarray, support: str = "support") -> dict:
    """The support headline on multi-record observations (support == 1: drawn by every record)."""
    sel = np.flatnonzero((classes == "D") | (classes == "M"))
    la = np.log(F["area"][sel].astype(float))
    strata_all = np.digitize(la, np.quantile(la, np.linspace(0, 1, 9)[1:-1]))
    multi = np.isfinite(F[support][sel])
    s, cl, kk, st = F[support][sel][multi], classes[sel][multi], k[sel][multi], strata_all[multi]
    allr = s == 1.0
    D, M = cl == "D", cl == "M"
    out = {"D_multi": int(D.sum()), "D_all_agree": int((D & allr).sum()), "M_multi": int(M.sum()),
           "D_all_agree_frac": float((D & allr).sum() / D.sum()), "M_all_agree_frac": float((M & allr).sum() / M.sum())}
    for kv in (2, 3):
        sk = kk == kv
        out[f"k{kv}_nD"], out[f"k{kv}_nM"] = int((D & sk).sum()), int((M & sk).sum())
        out[f"k{kv}_D_all_agree_frac"] = float((D & sk & allr).sum() / (D & sk).sum())
        out[f"k{kv}_M_all_agree_frac"] = float((M & sk & allr).sum() / (M & sk).sum())
    rw = 0.0
    for v in np.unique(st[D]):
        wD = (D & (st == v)).sum() / D.sum()
        mm = M & (st == v)
        rw += wD * (allr[mm].mean() if mm.any() else np.nan)
    out["M_all_agree_frac_area_reweighted_to_D"] = float(rw)
    return out


def summarise(R: list, slots: list) -> dict:
    R = [r for r in R if r["n"]]
    F = {k: np.concatenate([r["rows"][k] for r in R]) for k in R[0]["rows"]}
    stems = [r["case"] for r in R]
    img_id = np.concatenate([np.full(r["n"], i) for i, r in enumerate(R)])
    k = np.concatenate([np.full(r["n"], r["n_records"]) for r in R])
    classes = np.array([cls(m, p, t) for m, p, t in zip(F["matched"], F["p_max"], F["touched_by_deleted"],
                                                          strict=True)])
    cc = {c: int((classes == c).sum()) for c in "DABMC"}
    out = {"n_images_with_objects": len(stems), "n_objects": int(len(classes)), "class_counts": cc,
           "results": auc_block(F, classes, img_id), "headline": headline(F, classes, k), "per_model": {}}
    # negative controls: (i) a 50% support rule must move the all-record count (published 7 of 284, not 8);
    # (ii) classes shuffled within each image must lose the contrast AUC (D vs M only)
    out["negative_controls"] = {"support_50pct": headline(F, classes, k, "support50")}
    rng = np.random.default_rng(1)
    shuf = classes.copy()
    for i in np.unique(img_id):
        ix = np.flatnonzero(img_id == i)
        shuf[ix] = shuf[rng.permutation(ix)]
    out["negative_controls"]["shuffled_classes_D_contrast_auc"] = strat_auc(
        *_dm(F, shuf, "contrast"))
    D007_D = classes == "D"
    for nm in slots:
        pm = {"bags": []}
        for b in range(NBAG):
            A = {kk: np.concatenate([r["arms"][f"{nm}_b{b}"][kk] for r in R]) for kk in ("iou", "p_max",
                                                                                   "touched_by_deleted")}
            cb = np.array([cls(i > IOU_T, p, t) for i, p, t in zip(A["iou"], A["p_max"], A["touched_by_deleted"],
                                                                   strict=True)])
            rr = auc_block(F, cb, img_id)
            Dm = cb == "D"
            fin = np.isfinite(F["support"])
            pm["bags"].append({"class_counts": {c: int((cb == c).sum()) for c in "DABMC"}, "D": rr["D"],
                               "D_overlap_with_D007_D": int((Dm & D007_D).sum()),
                               "D_drawn_by_every_record": [int((Dm & fin & (F["support"] == 1.0)).sum()),
                                                           int((Dm & fin).sum())]})
        for ft in ("contrast", "support"):
            v = [bg["D"][ft]["auc"] for bg in pm["bags"]]
            pm[f"{ft}_AUC_bag_mean"] = float(np.mean(v))
            pm[f"{ft}_AUC_bag_range"] = [float(min(v)), float(max(v))]
        pm["n_D_bag_mean"] = float(np.mean([bg["class_counts"]["D"] for bg in pm["bags"]]))
        out["per_model"][nm] = pm
    out["_rows"] = {"cases": stems, "n_per_case": np.array([r["n"] for r in R]), "F": F, "classes": classes}
    return out


def run(cfgs: dict, d007_bin_dir: str, d007_prob_dir: str, workers: int = 3, records: str = REC.DEFAULT) -> dict:
    cfg0 = cfgs["FAST"]
    recs = REC.source(records).records_by_case(cfg0.train_case_map)
    fits = {f["stem"]: f for f in json.loads((fp.WORK / DISKFIT).read_text())["frames"]}
    sels = {nm: OS.build(cfg) for nm, cfg in cfgs.items()}
    cases = sels["FAST"]["cases"]
    items = []
    for i, c in enumerate(cases):
        slots = {}
        for nm, sel in sels.items():
            r = OS.rows_of_image(sel, i)
            slots[nm] = (cfgs[nm], [kk[r] for kk in sel["keeps"]], sel["area"][r])
        items.append((c, recs[c], records, f"{cfg0.raw_train}/{c}_0000.png", f"{d007_bin_dir}/{c}.png",
                      f"{d007_prob_dir}/{c}.png", fits[c], slots))
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            R = list(ex.map(image_rows, items, chunksize=2))
    else:
        R = [image_rows(it) for it in items]
    return summarise(sorted(R, key=lambda r: r["case"]), list(cfgs))
