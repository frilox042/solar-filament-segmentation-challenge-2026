"""Morphology of the kept (shipped-rule) masks on the 707 OOF observations, against the competition annotations.

ARMS        D007_400 / D007_all / D007_100: the D007 OOF argmax, 4-connected components >= 400 / >= 20 / >= 100 px
            (D007_100 is only a negative control); nnHALO_b*: the cross-fitted selector keeps (5 bags) of its
            candidate components; Tcand: every candidate, no selector.
OBJECT MORPHOLOGY  per 4-connected component (or annotation) >= 20 px, on its bounding-box crop padded by 3 px:
            skimage skeleton; arc = geodesic diameter of the 8-connected skeleton (double Dijkstra, edge weights 1 and
            sqrt 2); width = area / arc; branch pixels = skeleton pixels with > 2 skeleton neighbours (8-neighbourhood).
(a) WIDTH   pairs = every (annotation, object) hit (IoU > 0.5, per record); per image the mean of (object width -
            annotation width) over its pairs; a bagged arm averages the bags per image (NaN-aware); the mean over
            images, as a % of the median annotation width over D007_400's pairs (11.07 px). Also the mean absolute
            difference, and nnHALO - D007 on the annotations both match (paired bootstrap over images).
(b) BRANCHING  per image 1000 x branch pixels / skeleton pixels over the arm's objects, minus the same over the
            4-connected components >= 20 px of the union of every record's annotations >= 20 px; population = images
            where the reference and D007_all, D007_400, nnHALO are all defined; % of the reference median.
(c) SPLIT   per annotation: the number of kept objects covering >= 10% of it; per image the share of annotations
            with >= 2 among those with >= 1; mean over images. The FLOOR-deletion diagnostic: annotations split by the
            candidates, their mean covered fraction before and after the selector (bag 0), and their match rate.
(d) BARBS   per reference component: spine = the unweighted geodesic longest path of its skeleton, barb = the rest
            of the skeleton, tip = barb end points. Coverage = fraction of those pixels inside the arm's mask (per-image
            mean); "p < 0.3" = fraction outside the mask with map probability < 0.3 (pooled).
Bootstraps: mean of per-image differences, `default_rng(0)`, 5000 sequential draws (as in the reported computation).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse import csgraph

from filament_seg.config import SlotConfig
from filament_seg.evaluation import host_rows as HR
from filament_seg.evaluation import oof_selection as OS
from filament_seg.maps import load_bin, load_prob_png

N = 2048
PAD = 3
SQ2 = float(np.sqrt(2.0))
TAU_SPLIT = 0.10
TAU_BARB = 0.30
MIN_OBJ = 20
NBAG = 5
NB = 5000


# ---------------------------------------------------------------------------------------------- object morphology
def geodesic(sk: np.ndarray, weighted: bool = True, path: bool = False):
    """Double-sweep geodesic diameter of an 8-connected skeleton (start = first pixel in raster order). weighted: edge
    weights 1 / sqrt 2 (arc length), else hop count. path=True returns the longest-path pixels of every component."""
    ys, xs = np.nonzero(sk)
    n = len(ys)
    pos = -np.ones(sk.shape, np.int64)
    pos[ys, xs] = np.arange(n)
    h, w = sk.shape
    r, c, wt = [], [], []
    for dy, dx, e in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, SQ2), (1, -1, SQ2)):
        yy, xx = ys + dy, xs + dx
        ok = (yy < h) & (xx >= 0) & (xx < w)
        j = np.full(n, -1)
        j[ok] = pos[yy[ok], xx[ok]]
        m = j >= 0
        r.append(np.flatnonzero(m))
        c.append(j[m])
        wt.append(np.full(int(m.sum()), e if weighted else 1.0))
    r, c, wt = np.concatenate(r), np.concatenate(c), np.concatenate(wt)
    g = sparse.coo_matrix((np.r_[wt, wt], (np.r_[r, c], np.r_[c, r])), shape=(n, n)).tocsr()
    if not path:
        if n == 1:
            return 0.0
        d0 = csgraph.dijkstra(g, directed=False, indices=0)
        a = int(np.argmax(np.where(np.isfinite(d0), d0, -1)))
        d1 = csgraph.dijkstra(g, directed=False, indices=a)
        return float(np.max(np.where(np.isfinite(d1), d1, -1)))
    out = np.zeros(sk.shape, bool)
    nc, lab = csgraph.connected_components(g, directed=False)
    for k in range(nc):
        nodes = np.flatnonzero(lab == k)
        if len(nodes) == 1:
            out[ys[nodes[0]], xs[nodes[0]]] = True
            continue
        sub = g[nodes][:, nodes]
        d0 = csgraph.shortest_path(sub, unweighted=True, indices=0)
        u = int(np.argmax(np.where(np.isfinite(d0), d0, -1)))
        du, pred = csgraph.shortest_path(sub, unweighted=True, indices=u, return_predecessors=True)
        v = int(np.argmax(np.where(np.isfinite(du), du, -1)))
        while v != u and v >= 0:
            out[ys[nodes[v]], xs[nodes[v]]] = True
            v = pred[v]
        out[ys[nodes[u]], xs[nodes[u]]] = True
    return out


def neighbours8(sk: np.ndarray) -> np.ndarray:
    s = sk.astype(np.int16)
    return ndimage.convolve(s, np.ones((3, 3), np.int16), mode="constant") - s


def crop(flat: np.ndarray, pad: int = PAD) -> tuple[np.ndarray, int, int]:
    y, x = flat // N, flat % N
    y0, x0 = int(y.min()) - pad, int(x.min()) - pad
    M = np.zeros((int(y.max()) - y0 + 1 + pad, int(x.max()) - x0 + 1 + pad), bool)
    M[y - y0, x - x0] = True
    return M, y0, x0


def morph(flat: np.ndarray) -> tuple:
    """(area, arc, skeleton px, branch px, branch components) of one object (flat pixel indices); NaN below 20 px or
    for an empty skeleton."""
    from skimage.morphology import skeletonize
    a = float(len(flat))
    if a < MIN_OBJ:
        return (a, np.nan, np.nan, np.nan, np.nan)
    M, _, _ = crop(flat)
    sk = skeletonize(M)
    if not sk.any():
        return (a, np.nan, np.nan, np.nan, np.nan)
    bp = sk & (neighbours8(sk) > 2)
    return (a, geodesic(sk), float(sk.sum()), float(bp.sum()), float(ndimage.label(bp)[1]) if bp.any() else 0.0)


def flat_lists(lab: np.ndarray, n: int) -> tuple[list, np.ndarray]:
    """Flat pixel indices of labels 1..n (label order) and their areas."""
    fl = lab.ravel()
    o = np.argsort(fl, kind="stable")
    cnt = np.bincount(fl, minlength=n + 1)
    st = np.r_[0, np.cumsum(cnt)]
    return [o[st[k]:st[k + 1]] for k in range(1, n + 1)], cnt[1:]


def width(m: tuple) -> float:
    return m[0] / m[1] if np.isfinite(m[1]) and m[1] > 1e-6 else np.nan


# ---------------------------------------------------------------------------------------------- per image
def image_worker(item: tuple) -> dict:
    """item = (i, case, records' segmentations, {"nnHALO": (cfg, keeps, area)}, D007 png, nc_t)."""
    from threadpoolctl import threadpool_limits
    i, case, segs, slots, d007_png, nc_t = item
    with threadpool_limits(1):
        return _image(i, case, segs, slots, d007_png, nc_t)


def _image(i, case, segs, slots, d007_png, nc_t):
    recs = [[HR.decode(s) for s in rec] for rec in segs]
    gt20 = np.zeros(N * N, bool)
    for r in recs:
        for ix in r:
            if len(ix) >= MIN_OBJ:
                gt20[ix] = True
    S = {}
    lab, n = ndimage.label(load_bin(d007_png))
    S["B"] = {"lab": lab.astype(np.int32), "n": n, "pr": load_prob_png(f"{slots['nnHALO'][0].oof_prob_dir}/{case}.png")}
    out = {"i": i, "case": case, "k": len(recs)}
    for src, nm in (("T", "nnHALO"),):
        cfg, keeps, rows_area = slots[nm]
        cl, area = OS.candidate_labels(cfg, case)
        assert np.array_equal(area, rows_area), f"candidate areas != OOF rows ({case})"
        S[src] = {"lab": cl, "n": len(area), "pr": load_prob_png(f"{cfg.oof_prob_dir}/{case}.png"), "keeps": keeps}
        if nc_t is not None:
            # negative control of the area gate: the base at a different threshold gives different candidates
            from dataclasses import replace
            _, a2 = OS.candidate_labels(replace(cfg, core_t=nc_t), case)
            out["nc_t_candidates_equal"] = bool(np.array_equal(a2, rows_area))
    for s in S.values():
        s["flat"], s["area"] = flat_lists(s["lab"], s["n"])
    aB, nT = S["B"]["area"], S["T"]["n"]
    arms = {"D007_400": ("B", aB >= 400), "D007_100": ("B", aB >= 100), "D007_all": ("B", aB >= MIN_OBJ),
            "Tcand": ("T", np.ones(nT, bool))}
    for b in range(NBAG):
        arms[f"nnHALO_b{b}"] = ("T", S["T"]["keeps"][b])
    need = {s: np.zeros(S[s]["n"], bool) for s in S}
    for s, k in arms.values():
        need[s] |= k
    mo = {s: {int(c): morph(S[s]["flat"][c]) for c in np.flatnonzero(need[s])} for s in S}
    # ---- annotation x arm: hit pairs and split rows
    pairs = {a: [] for a in arms}
    split = {a: [] for a in arms}
    annmo = {}
    for r, rec in enumerate(recs):
        for j, ix in enumerate(rec):
            na = len(ix)
            Js = {s: np.bincount(S[s]["lab"].ravel()[ix], minlength=S[s]["n"] + 1)[1:] if na
                  else np.zeros(S[s]["n"], np.int64) for s in S}
            for a, (s, k) in arms.items():
                if na == 0:
                    split[a].append((r, j, 0, False, 0.0))
                    continue
                Jk = np.where(k, Js[s], 0)
                iou = np.where(Jk > 0, Jk / (S[s]["area"] + na - Jk), 0.0)
                h = np.flatnonzero(iou > 0.5)
                for c in h:
                    pairs[a].append((r, j, int(c), float(iou[c])))
                    if (r, j) not in annmo:
                        annmo[(r, j)] = morph(ix)
                split[a].append((r, j, int((Jk / na >= TAU_SPLIT).sum()), bool(len(h)), float(Jk.sum() / na)))
    # width / branch tables per arm
    out["pairs"] = {}
    for a, P in pairs.items():
        s = arms[a][0]
        rows = []
        for r, j, c, _ in P:
            m, am = mo[s][c], annmo[(r, j)]
            rows.append((r, j, width(m), width(am), m[3], m[2], am[3], am[2]))
        out["pairs"][a] = np.array(rows, float).reshape(-1, 8)
    out["split"] = {a: np.array(v, float).reshape(-1, 5) for a, v in split.items()}
    out["branch"] = {}
    for a, (s, k) in arms.items():
        sk = bp = 0.0
        for c in np.flatnonzero(k):
            m = mo[s][int(c)]
            if np.isfinite(m[2]) and m[2] > 0:
                sk += m[2]
                bp += m[3]
        out["branch"][a] = (bp, sk)
    # ---- reference (>= 20 px annotation union) morphology and barb skeleton
    from skimage.morphology import skeletonize
    gl, gn = ndimage.label(gt20.reshape(N, N))
    gfl, _ = flat_lists(gl.astype(np.int32), gn)
    gm = [morph(f) for f in gfl if len(f) >= MIN_OBJ]
    ok = [m for m in gm if np.isfinite(m[2]) and m[2] > 0]
    out["gt_branch"] = (sum(m[3] for m in ok), sum(m[2] for m in ok))
    pts = {"spine": [], "barb": [], "tip": []}
    for f in gfl:
        if len(f) < MIN_OBJ:
            continue
        M, y0, x0 = crop(f)
        sk = skeletonize(M)
        if not sk.any():
            continue
        sp = geodesic(sk, weighted=False, path=True)
        barb = sk & ~sp
        tip = barb & (neighbours8(sk) == 1)
        for nm, m in (("spine", sp), ("barb", barb), ("tip", tip)):
            yy, xx = np.nonzero(m)
            yy, xx = yy + y0, xx + x0
            v = (yy >= 0) & (yy < N) & (xx >= 0) & (xx < N)
            pts[nm].append(yy[v] * N + xx[v])
    pts = {k: (np.concatenate(v) if v else np.zeros(0, np.int64)) for k, v in pts.items()}
    out["barb"] = {}
    for a, (s, k) in arms.items():
        km = np.r_[False, k]
        lab, pr = S[s]["lab"].ravel(), S[s]["pr"].ravel()
        out["barb"][a] = {}
        for nm, ix in pts.items():
            inm, lo = km[lab[ix]], pr[ix] < TAU_BARB
            out["barb"][a][nm] = (int(inm.sum()), int((~inm & ~lo).sum()), int((~inm & lo).sum()), int(len(ix)))
    return out


# ---------------------------------------------------------------------------------------------- statistics
def pboot(a, b, nb: int = NB, seed: int = 0) -> dict:
    """Mean paired difference over finite pairs; CI from nb sequential resamples of one generator."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    rng = np.random.default_rng(seed)
    m = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(nb)])
    lo, hi = np.percentile(m, [2.5, 97.5])
    return {"mean_diff": float(d.mean()), "ci95": [float(lo), float(hi)], "n": int(len(d))}


def nanmean_rows(M) -> np.ndarray:
    """Column-wise mean over the finite entries; NaN where a column has none (no warning)."""
    import warnings
    M = np.asarray(M, float)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.where(np.isfinite(M).any(0), np.nanmean(np.where(np.isfinite(M), M, np.nan), 0), np.nan)


def _bags(m: str) -> list[str]:
    return [f"{m}_b{b}" for b in range(NBAG)] if m == "nnHALO" else [m]


def summarise(R: list) -> dict:
    R = sorted(R, key=lambda r: r["i"])
    n = len(R)
    out: dict = {"n_images": n}

    # (a) width
    def own(arm):
        sg, ab, wr = np.full(n, np.nan), np.full(n, np.nan), []
        for i, r in enumerate(R):
            T = r["pairs"][arm]
            ok = np.isfinite(T[:, 2]) & np.isfinite(T[:, 3]) if len(T) else np.zeros(0, bool)
            if ok.any():
                d = T[ok, 2] - T[ok, 3]
                sg[i], ab[i] = d.mean(), np.abs(d).mean()
                wr.extend(T[ok, 3].tolist())
        return sg, ab, wr

    W = {}
    for m in ("D007_400", "D007_100", "nnHALO"):
        res = [own(a) for a in _bags(m)]
        W[m] = (nanmean_rows([x[0] for x in res]), nanmean_rows([x[1] for x in res]), res[0][2])
    med = float(np.median(W["D007_400"][2]))
    out["width"] = {"denominator_px": med, **{m: {"signed_px": float(np.nanmean(sg)),
                                                  "pct": 100 * float(np.nanmean(sg)) / med,
                                                  "abs_px": float(np.nanmean(ab)), "n_img": int(np.isfinite(sg).sum())}
                                              for m, (sg, ab, _) in W.items()}}

    def common(arm, ref="D007_400"):
        A, B = np.full(n, np.nan), np.full(n, np.nan)
        for i, r in enumerate(R):
            X, Y = r["pairs"][arm], r["pairs"][ref]
            if not len(X) or not len(Y):
                continue
            kx = {(int(a), int(b)): k for k, (a, b) in enumerate(X[:, :2])}
            ky = {(int(a), int(b)): k for k, (a, b) in enumerate(Y[:, :2])}
            ks = sorted(set(kx) & set(ky))
            if not ks:
                continue
            x, y = X[[kx[k] for k in ks]], Y[[ky[k] for k in ks]]
            ok = np.isfinite(x[:, 2]) & np.isfinite(y[:, 2]) & np.isfinite(x[:, 3])
            if ok.any():
                A[i], B[i] = (x[ok, 2] - x[ok, 3]).mean(), (y[ok, 2] - y[ok, 3]).mean()
        return A, B

    for m in ("nnHALO",):
        cs = [common(a) for a in _bags(m)]
        out["width"][f"{m}-D007_400_same_annotations_px"] = pboot(nanmean_rows([c[0] for c in cs]),
                                                                  nanmean_rows([c[1] for c in cs]))

    # (b) branching
    def rate(bp_sk):
        bp, sk = bp_sk
        return 1000 * bp / sk if sk > 0 else np.nan
    g = np.array([rate(r["gt_branch"]) for r in R])
    V = {m: nanmean_rows([[rate(r["branch"][a]) for r in R] for a in _bags(m)])
         for m in ("D007_all", "D007_400", "nnHALO")}
    pop = np.isfinite(g) & np.all([np.isfinite(v) for v in V.values()], axis=0)
    gmed = float(np.median(g[pop]))
    out["branching"] = {"n_pop": int(pop.sum()), "reference_median_per_1k": gmed}
    for m, v in V.items():
        bt = pboot(v[pop], g[pop])
        out["branching"][m] = {"pct": 100 * bt["mean_diff"] / gmed, "pct_ci": [100 * x / gmed for x in bt["ci95"]],
                               "n": bt["n"]}

    # (c) split degree
    def split_img(arm):
        ff = np.full(n, np.nan)
        for i, r in enumerate(R):
            F = r["split"][arm]
            m = F[:, 2] >= 1
            if m.any():
                ff[i] = (F[m, 2] >= 2).mean()
        return ff
    out["split"] = {m: float(np.nanmean(nanmean_rows([split_img(a) for a in _bags(m)])))
                    for m in ("D007_400", "D007_100", "nnHALO", "Tcand")}
    dele = {}
    for m, cand in (("nnHALO", "Tcand"),):
        cov_c, cov_k, mc, mk = [], [], [], []
        for r in R:
            Fc, Fk = r["split"][cand], r["split"][f"{m}_b0"]
            sel = Fc[:, 2] >= 2
            cov_c.extend(Fc[sel, 4])
            cov_k.extend(Fk[sel, 4])
            mc.extend(Fc[sel, 3])
            mk.extend(Fk[sel, 3])
        dele[m] = {"n_annotations_split_by_candidates": len(cov_c), "mean_coverage_candidates": float(np.mean(cov_c)),
                   "mean_coverage_kept": float(np.mean(cov_k)), "matched_frac_candidates": float(np.mean(mc)),
                   "matched_frac_kept": float(np.mean(mk))}
    out["selector_deletion_bag0"] = dele

    # (d) barbs
    def pooled_lowp(arm, k):
        t = np.sum([r["barb"][arm][k] for r in R], axis=0)
        return float(t[2] / t[3])

    def img_cov(arm, k):
        return np.array([r["barb"][arm][k][0] / r["barb"][arm][k][3] if r["barb"][arm][k][3] else np.nan for r in R])
    out["barbs"] = {"barb_px": int(np.sum([r["barb"]["D007_all"]["barb"][3] for r in R])),
                    "outside_p_lt_0.3": {"D007_all": pooled_lowp("D007_all", "barb")}}
    for k in ("spine", "barb"):
        C = {m: nanmean_rows([img_cov(a, k) for a in _bags(m)]) for m in ("D007_400", "nnHALO")}
        out["barbs"][f"{k}_coverage_per_image_mean"] = {m: float(np.nanmean(v)) for m, v in C.items()}
        for m in ("nnHALO",):
            out["barbs"][f"{k}_coverage_{m}-D007_400"] = pboot(C[m], C["D007_400"])
    nc = [r["nc_t_candidates_equal"] for r in R if "nc_t_candidates_equal" in r]
    out["negative_control_threshold_candidates_equal"] = nc
    return out


def run(cfgs: dict[str, SlotConfig], d007_dir: str, workers: int, nc_t: float = 0.92, nc_every: int = 50) -> dict:
    """Every OOF observation (parallel), then the summary."""
    from concurrent.futures import ProcessPoolExecutor
    any_cfg = cfgs["nnHALO"]
    recs = HR.records_by_case(any_cfg.coco, any_cfg.train_case_map)
    sels = {nm: OS.build(cfg) for nm, cfg in cfgs.items()}
    cases = sels["nnHALO"]["cases"]
    items = []
    for i, c in enumerate(cases):
        slots = {}
        for nm, sel in sels.items():
            r = OS.rows_of_image(sel, i)
            slots[nm] = (cfgs[nm], [k[r] for k in sel["keeps"]], sel["area"][r])
        items.append((i, c, [s for _, s in recs[c]], slots, f"{d007_dir}/{c}.png", nc_t if i % nc_every == 0 else None))
    with ProcessPoolExecutor(workers) as ex:
        R = list(ex.map(image_worker, items, chunksize=2))
    return summarise(R)
