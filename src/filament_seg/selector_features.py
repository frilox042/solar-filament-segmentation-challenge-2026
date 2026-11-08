"""Selector features: per-object inputs of the FLOOR selector and its 22-column design matrix (k = 2 members).

The same functions build the 707 out-of-fold training rows and the 180 test rows, so the two cannot drift. That
includes the member masks (`member_masks`): a file member is read from its PNG, the derived member P is computed from
the very probability map the candidates come from.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from filament_seg.candidates import label_base, small_components

BASE_FEATS = ["area", "p_mean", "p_p10", "p_min", "p_ring", "margin", "frac_lo", "fill", "aspect"]
CTX_FEATS = ["n_comp", "img_p_mean", "area_rank", "nn_dist", "r_over_R"]


def member_masks(members, pr: np.ndarray, path_of) -> list:
    """The bool member masks of one image, in column order. `members`: `config.Member`s; `pr`: the base probability
    map (float32) of that image; `path_of(member)`: the PNG of a file member for that image.
    kind "file" -> `maps.load_bin(path)`; kind "base_threshold" -> `pr >= t` (t = 0.5 for P)."""
    from filament_seg.maps import load_bin
    out = []
    for m in members:
        if m.kind == "file":
            out.append(load_bin(path_of(m)))
        elif m.kind == "base_threshold":
            out.append(pr >= m.t)
        else:
            raise ValueError(m.kind)
    return out


def joint_iou(pl: np.ndarray, pn: int, pa: np.ndarray, gl: np.ndarray, gn: int) -> tuple[np.ndarray, np.ndarray]:
    """(pn, gn) IoU matrix of the components of label map pl (areas pa) against label map gl; and gl's areas."""
    J = np.bincount(pl.ravel().astype(np.int64) * (gn + 1) + gl.ravel(),
                    minlength=(pn + 1) * (gn + 1)).reshape(pn + 1, gn + 1)[1:, 1:]
    ga = np.bincount(gl.ravel(), minlength=gn + 1)[1:]
    return np.where(J > 0, J / np.maximum(pa[:, None] + ga[None, :] - J, 1), 0.0), ga


def object_features(L: np.ndarray, n: int, pa: np.ndarray, p: np.ndarray, pr: np.ndarray, raw,
                    member_masks: list) -> tuple:
    """Per-object selector inputs for the n components of label map L (areas pa) of base mask p.

    Returns (mb, cov, feats, ctx):
      mb, cov  (n, k)  best IoU against any component of member j, and the fraction of the object member j covers
      feats    (n, 9)  BASE_FEATS: area, mean / p10 / min probability inside, mean in an 8-px ring, margin, fraction
                       of pixels < 0.9, bbox fill, bbox aspect
      ctx      (n, 5)  CTX_FEATS: component count, image mean probability on p, area rank, nearest-neighbour
                       centroid distance, distance to the solar-disk centre / disk radius
    member_masks: bool masks in column order; raw: channel-0 image (PIL image or array)."""
    flat = L.ravel()
    # ---- member features: best IoU of each base component vs any member component, and coverage
    mb, cov = np.zeros((n, len(member_masks))), np.zeros((n, len(member_masks)))
    for j, mm in enumerate(member_masks):
        if n:
            cov[:, j] = np.bincount(flat, weights=mm.ravel(), minlength=n + 1)[1:] / pa
            ml, mn = ndimage.label(mm)
            if mn:
                mb[:, j] = joint_iou(L, n, pa, ml, mn)[0].max(1)
    # ---- the nine base features (bbox crop, 8-iteration ring inside the crop), and image context
    feats = np.zeros((n, len(BASE_FEATS)))
    objs = ndimage.find_objects(L)
    cent = np.zeros((n, 2))
    for i, sl in enumerate(objs):
        sm = L[sl] == (i + 1)
        sp = pr[sl]
        ring = ndimage.binary_dilation(sm, iterations=8) & ~sm
        ins = sp[sm]
        pring = float(sp[ring].mean()) if ring.any() else 0.0
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        feats[i] = [int(sm.sum()), float(ins.mean()), float(np.percentile(ins, 10)), float(ins.min()),
                    pring, float(ins.mean() - pring), float((ins < 0.9).mean()),
                    float(sm.sum() / max(1, h * w)), float(max(h, w) / max(1, min(h, w)))]
        ys, xs = np.nonzero(sm)
        cent[i] = [ys.mean() + sl[0].start, xs.mean() + sl[1].start]
    raw = np.asarray(raw).astype(np.float32)
    disk = raw / max(raw.max() / 255.0, 1e-6) > 40
    dl, dn = ndimage.label(disk)
    if dn:
        big = np.bincount(dl.ravel())[1:].argmax() + 1
        disk = ndimage.binary_fill_holes(dl == big)
        yy, xx = np.nonzero(disk)
        cy, cx, R = yy.mean(), xx.mean(), np.sqrt(disk.sum() / np.pi)
    else:
        cy, cx, R = 1024.0, 1024.0, 1000.0
    ctx = np.zeros((n, len(CTX_FEATS)))
    if n:
        img_p = float(pr[p].mean()) if p.any() else 0.0
        order = np.argsort(-pa)
        rank = np.empty(n)
        rank[order] = np.arange(n) / max(n - 1, 1)
        if n > 1:
            dd = np.sqrt(((cent[:, None, :] - cent[None, :, :]) ** 2).sum(-1))
            np.fill_diagonal(dd, np.inf)
            nn = dd.min(1)
        else:
            nn = np.full(n, 3000.0)
        rr = np.sqrt((cent[:, 0] - cy) ** 2 + (cent[:, 1] - cx) ** 2) / R
        ctx[:] = np.stack([np.full(n, n), np.full(n, img_p), rank, nn, rr], 1)
    return mb, cov, feats, ctx


def floor_small_rows(p: np.ndarray, pr: np.ndarray, raw, members: list, L: np.ndarray, n: int) -> tuple:
    """The 100-299 px components of p and their inputs, with the FLOOR image-context override: n_comp = the image's
    >= 300 px count n; nn_dist = distance to the nearest >= 300 px centroid (3000 if none). (L, n) = label_base(p).
    Returns (Ls, ns, sa, mb, cov, feats, ctx)."""
    Ls, ns, sa, _, _ = small_components(p)
    mb, cov, feats, ctx = object_features(Ls, ns, sa, p, pr, raw, members)
    if ns:
        ctx[:, 0] = n
        bc = np.array(ndimage.center_of_mass(np.ones_like(L), L, range(1, n + 1))) if n else np.zeros((0, 2))
        cc = np.array(ndimage.center_of_mass(np.ones_like(Ls), Ls, range(1, ns + 1)))
        ctx[:, 3] = (np.sqrt(((cc[:, None, :] - bc[None, :, :]) ** 2).sum(-1)).min(1) if n
                     else np.full(ns, 3000.0))
    return Ls, ns, sa, mb, cov, feats, ctx


def step2_matrix(feats: np.ndarray, mb: np.ndarray, cov: np.ndarray, ctx: np.ndarray) -> np.ndarray:
    """Design matrix: feats | mb | cov | votes (#mb > 0.5) | mean mb | min mb | min cov | ctx."""
    return np.column_stack([feats, mb, cov, (mb > 0.5).sum(1), mb.mean(1), mb.min(1), cov.min(1), ctx])


def image_rows(p: np.ndarray, pr: np.ndarray, raw, members: list) -> dict:
    """Both row populations of one image: big (>= 300 px, via label_base) and small (100-299 px, FLOOR)."""
    L, n, pa, _ = label_base(p)
    mb, cov, feats, ctx = object_features(L, n, pa, p, pr, raw, members)
    Ls, ns, sa, smb, scov, sfeats, sctx = floor_small_rows(p, pr, raw, members, L, n)
    return dict(L=L, n=n, pa=pa, mb=mb, cov=cov, feats=feats, ctx=ctx,
                Ls=Ls, ns=ns, sa=sa, smb=smb, scov=scov, sfeats=sfeats, sctx=sctx)
