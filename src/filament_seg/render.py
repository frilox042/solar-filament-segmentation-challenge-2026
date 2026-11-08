"""Apply the selector to the 180 test images and render the submission: kept components -> PNGs -> RLE CSV -> verify.

Per test image: probability map (the D007 fold-0 npz, quantised like the OOF maps) -> base mask -> the same rows as the
training side (members: the D036 test binary, and P = that same quantised map >= 0.5) -> keep a component iff the bag's
mean prediction > cut.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

from filament_seg.candidates import base_mask, label_base, small_components
from filament_seg.config import SlotConfig
from filament_seg.maps import load_bin, npz_foreground, quantise_like_oof
from filament_seg.selector_features import image_rows, member_masks, step2_matrix
from filament_seg.submission import rle, verify


def test_setup(cfg: SlotConfig) -> tuple[list, dict, dict]:
    """(stems: the 180 competition image ids sorted, inv: image id -> case id, mf: image id -> the members' test PNGs,
    one entry per member in column order, None for a member computed in process)."""
    cm = json.load(open(cfg.test_case_map))["test"]
    inv = {os.path.splitext(v if isinstance(v, str) else v["image"])[0]: k for k, v in cm.items()}
    stems = sorted(inv)
    assert len(stems) == 180, len(stems)
    mf = {s: [m.test_file(inv[s], s) if m.is_file else None for m in cfg.members] for s in stems}
    miss = [f for s in stems for f in mf[s] if f is not None and not os.path.exists(f)]
    assert not miss, f"{len(miss)} member files missing, e.g. {miss[:3]}"
    if cfg.member_join_check:
        # a silently failed join (wrong naming) would open the wrong image: each file member must overlap the
        # D007 >= 0.5 mask on the first 8 images (IoU >= 0.20)
        for s in stems[:8]:
            b = npz_foreground(f"{cfg.test_prob_dir}/{inv[s]}.npz") >= 0.5
            for f in mf[s]:
                if f is None:
                    continue
                m = load_bin(f)
                j = (b & m).sum() / max((b | m).sum(), 1)
                assert j >= 0.20, f"member join failure: {f} IoU {j:.3f} vs the base"
    return stems, inv, mf


def test_prob(cfg: SlotConfig, cid: str) -> np.ndarray:
    """The float32 probability map of one test case (the base network's npz, quantised like the OOF maps)."""
    return quantise_like_oof(npz_foreground(f"{cfg.test_prob_dir}/{cid}.npz"))


def test_rows(args: tuple) -> dict:
    """One test image's rows. args = (cfg, stem, case id, member files: `test_setup`'s mf[stem])."""
    from threadpoolctl import threadpool_limits
    cfg, stem, cid, member_files = args
    files = dict(zip([m.name for m in cfg.members], member_files, strict=True))
    with threadpool_limits(1):
        pr = test_prob(cfg, cid)
        p = base_mask(cfg.base_mask, pr, cfg.core_t)
        members = member_masks(cfg.members, pr, lambda m: files[m.name])
        raw = Image.open(f"{cfg.raw_test}/{cid}_0000.png")
        r = image_rows(p, pr, raw, members)
    k = len(members)
    return dict(stem=stem, n=r["n"], ns=r["ns"], feats=r["feats"].reshape(-1, 9), mb=r["mb"].reshape(-1, k),
                cov=r["cov"].reshape(-1, k), ctx=r["ctx"].reshape(-1, 5), sfeats=r["sfeats"].reshape(-1, 9),
                smb=r["smb"].reshape(-1, k), scov=r["scov"].reshape(-1, k), sctx=r["sctx"].reshape(-1, 5))


def design_matrix(rows: list) -> tuple:
    """Stack per-image rows as [all >= 300 px rows | all 100-299 px rows].
    Returns (X, area, image index per row, image index per big row, image index per small row)."""
    timg = np.concatenate([np.full(r["n"], i) for i, r in enumerate(rows)]).astype(int)
    simg = np.concatenate([np.full(r["ns"], i) for i, r in enumerate(rows)]).astype(int)

    def cat(k):
        return np.concatenate([r[k] for r in rows])

    X = np.vstack([step2_matrix(cat("feats"), cat("mb"), cat("cov"), cat("ctx")),
                   step2_matrix(cat("sfeats"), cat("smb"), cat("scov"), cat("sctx"))])
    area = np.r_[cat("feats")[:, 0], cat("sfeats")[:, 0]]
    return X, area, np.r_[timg, simg], timg, simg


def scores(bag: list, X: np.ndarray) -> np.ndarray:
    """Each candidate's score: the bag's mean prediction of its expected per-record PQ credit."""
    return np.mean([m.predict(X) for m in bag], 0)


def select(bag: list, X: np.ndarray, cut: float) -> np.ndarray:
    """The keep rule: score strictly above the cut."""
    return scores(bag, X) > cut


def kept_mask(p: np.ndarray, keep_big: np.ndarray, keep_small: np.ndarray) -> np.ndarray:
    """The output mask of one image: its kept >= 300 px and kept 100-299 px components of base mask p."""
    L, n, _, _ = label_base(p)
    _, _, _, pl, small_ids = small_components(p)
    assert n == len(keep_big) and len(small_ids) == len(keep_small)
    return np.isin(L, np.flatnonzero(keep_big) + 1) | np.isin(pl, small_ids[keep_small])


def apply(cfg: SlotConfig, bag: list, cut: float, csv_path, verify_json=None) -> dict:
    """Score, keep, write PNGs (cfg.scratch/by_image), the RLE CSV, and run the verify gate."""
    stems, inv, mf = test_setup(cfg)
    with ProcessPoolExecutor(cfg.apply_workers) as ex:
        tr = list(ex.map(test_rows, [(cfg, s, inv[s], mf[s]) for s in stems], chunksize=4))
    Xt, tarea, oimg, timg, simg = design_matrix(tr)
    keep = select(bag, Xt, cut)
    png_dir = cfg.scratch / "by_image"
    os.makedirs(png_dir, exist_ok=True)
    for f in glob.glob(f"{png_dir}/*.png"):
        os.remove(f)
    ko300, kos = keep[:len(timg)], keep[len(timg):]
    for i, s in enumerate(stems):
        p = base_mask(cfg.base_mask, test_prob(cfg, inv[s]), cfg.core_t)
        m = kept_mask(p, ko300[timg == i], kos[simg == i])
        assert int(m.sum()) == int(tarea[keep & (oimg == i)].sum())
        Image.fromarray(m.astype(np.uint8) * 255).save(f"{png_dir}/{s}.png")
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    rle.write_csv(png_dir, csv_path, cfg.min_area_csv)
    vrc = None
    if verify_json:
        try:
            vrc = verify.main([str(csv_path), str(png_dir), "--min-area", str(cfg.min_area_csv),
                               "--json", str(verify_json)])
        except SystemExit as e:
            vrc = e.code
        vrc = int(vrc or 0)
    raw_csv = open(csv_path, "rb").read()
    n_img_keep = np.bincount(oimg[keep], minlength=len(stems))
    return dict(n_rows=raw_csv.count(b"\n") - 1, md5=hashlib.md5(raw_csv).hexdigest(), kept_px=int(tarea[keep].sum()),
                silenced=[stems[i] for i in range(len(stems)) if n_img_keep[i] == 0], verify_rc=vrc)
