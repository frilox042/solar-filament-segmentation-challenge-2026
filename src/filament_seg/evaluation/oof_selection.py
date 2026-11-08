"""Cross-fitted selector keeps on the 707 out-of-fold (OOF) training rows, and the kept OOF masks they define.

Every evaluation number scores the selector OUT OF SAMPLE. The deployed bag (`selector_fit.fit`) is trained on all
707 images, so it cannot score them. `crossfit` instead splits the images into 5 outer folds (seed 1000 + s) and,
for each fold f:

  cut_f   = PQ_w05(base components >= 400 px, images outside f) / 2
  bag_f   = selector_fit.fit_bag on the rows of the images outside f (same HGB settings as the deployed bag)
  uhat    = mean of bag_f's 5 predictions on the rows of f;  keep = uhat > cut_f

Five seeds s = 0..4 give five "bags" of keeps; every pooled statistic is the mean over the five (the cross-fit used by
every selector result quoted in the paper).

Row order is `selector_fit.floor_table`'s: all >= 300 px rows image by image, then all 100-299 px rows image by
image. Within one image the rows are therefore [its >= 300 px components in label order, then its 100-299 px
components in label order], which is what `candidate_labels` paints.

    python scripts/eval_oof_selection.py     (cached in WORK/eval/)

The rows come from `selector_fit.build_oof_table` (the OOF maps, member masks and channel 0), or, when WORK was staged
from the evaluation bundle (`filament_seg.evaluation.bundle`), from the bundled rows `WORK/analysis/oof_rows_<slot>.npz`
(the same arrays, saved by `bundle.export`).
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np

from filament_seg import filament_paths as fp
from filament_seg.candidates import base_mask, label_base, small_components
from filament_seg.config import SlotConfig
from filament_seg.evaluation import bundle
from filament_seg.maps import load_prob_png
from filament_seg.scorer import Ref
from filament_seg.selector_fit import build_oof_table, fit_bag, image_folds

SEEDS = (0, 1, 2, 3, 4)
OUTER_SEED0 = 1000
BASE_MIN = 400                         # the base whose PQ defines the cut


def eval_dir() -> Path:
    return Path(os.environ.get("FILAMENT_EVAL", fp.WORK / "eval"))


def crossfit(Zc: dict, X: np.ndarray, area: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, list]:
    """One seed: (keep bool over rows, uhat over rows, the 5 per-fold cuts)."""
    n_img = len(Zc["cases"])
    fo = image_folds(n_img, OUTER_SEED0 + seed)
    rows = np.ones(len(area), bool)
    base = area >= BASE_MIN
    R, Rb = Ref(Zc, "coco", rows), Ref(Zc, "coco", base)
    keep, uhat, cuts = np.zeros(len(area), bool), np.full(len(area), np.nan), []
    for f in range(5):
        tr = fo[R.img] != f
        cut = Rb.score(fo[Rb.img] != f, fo != f)["pq_w05"] / 2
        bag = fit_bag(X[tr], R.u[tr], R.img[tr], base[tr])
        uh = np.mean([m.predict(X[~tr]) for m in bag], 0)
        keep[~tr], uhat[~tr] = uh > cut, uh
        cuts.append(float(cut))
    return keep, uhat, cuts


def input_identity(cfg: SlotConfig, n_sample: int = 5) -> dict:
    """What the cached keeps were computed from: the slot config file and the bytes of a sample of the OOF inputs
    (the base map and every member mask of n_sample cases spread over the case list). A retrain that writes new maps
    under the same names changes the bytes, so a stale cache is refused instead of silently re-used.
    From a staged evaluation bundle (`bundle.rows_path` exists): the config and the sha256 of the bundled rows."""
    import hashlib
    import json
    rows = bundle.rows_path(cfg.slot)
    if rows.exists():
        conf = hashlib.sha256((fp.CONFIGS / f"{cfg.slot}.json").read_bytes()).hexdigest()
        return {"slot": cfg.slot, "config_sha256": conf, "source": "eval_bundle",
                "oof_rows_sha256": bundle.sha256(rows)}
    cases = sorted(json.load(open(cfg.train_case_map)).get("train", {}))
    pick = [cases[i] for i in np.linspace(0, len(cases) - 1, n_sample).astype(int)] if cases else []
    h = hashlib.sha256()
    for c in pick:
        for d in (cfg.oof_prob_dir, *cfg.oof_member_dirs):
            h.update(Path(f"{d}/{c}.png").read_bytes())
    conf = hashlib.sha256((fp.CONFIGS / f"{cfg.slot}.json").read_bytes()).hexdigest()
    return {"slot": cfg.slot, "config_sha256": conf, "sample_cases": pick, "sample_inputs_sha256": h.hexdigest()}


def build(cfg: SlotConfig, threads: int = 4, cache: bool = True, rebuild: bool = False) -> dict:
    """{cases, Z (row accounting table), area, keeps[5], uhat[5], cuts[5], identity} for the slot; cached per slot.
    A cache whose `input_identity` differs from the current inputs is refused (pass rebuild=True to recompute)."""
    from threadpoolctl import threadpool_limits
    path = eval_dir() / f"{cfg.slot}_oof_selection.pkl"
    ident = input_identity(cfg)
    if cache and path.exists() and not rebuild:
        sel = pickle.load(open(path, "rb"))
        if sel.get("identity") != ident:
            raise RuntimeError(f"{path} was built from other inputs ({sel.get('identity')} != {ident}); "
                               "re-run scripts/eval_oof_selection.py --rebuild")
        return sel
    rows = bundle.rows_path(cfg.slot)
    Zc, X, area = bundle.load_rows(rows) if rows.exists() else build_oof_table(cfg)
    with threadpool_limits(threads):
        res = [crossfit(Zc, X, area, s) for s in SEEDS]
    out = {"slot": cfg.slot, "identity": ident, "cases": [str(c) for c in Zc["cases"]],
           "Z": {k: np.asarray(v) for k, v in Zc.items()}, "area": np.asarray(area),
           "keeps": [r[0] for r in res], "uhat": [r[1] for r in res], "cuts": [r[2] for r in res]}
    if cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(out, open(f"{path}.tmp", "wb"))
        os.replace(f"{path}.tmp", path)
    return out


def pq_per_bag(sel: dict, imgs: np.ndarray | None = None) -> list[dict]:
    """Per-record host PQ (w 0.5 / 1.0, pooled over images) of each bag's keeps."""
    R = Ref(sel["Z"], "coco", np.ones(len(sel["area"]), bool))
    return [R.score(k, imgs) for k in sel["keeps"]]


def rows_of_image(sel: dict, i: int) -> np.ndarray:
    """Global row indices of image i, in candidate-label order (>= 300 px rows, then 100-299 px rows)."""
    return np.flatnonzero(sel["Z"]["img"] == i)


def candidate_labels(cfg: SlotConfig, case: str) -> tuple[np.ndarray, np.ndarray]:
    """(label image, areas) of one OOF image's candidates in row order: label j = row j of that image (1-based)."""
    pr = load_prob_png(f"{cfg.oof_prob_dir}/{case}.png")
    p = base_mask(cfg.base_mask, pr, cfg.core_t)
    L, n, pa, _ = label_base(p)
    Ls, ns, sa, _, _ = small_components(p)
    cl = L.astype(np.int32)
    cl[Ls > 0] = (Ls[Ls > 0] + n).astype(np.int32)
    return cl, np.r_[pa, sa].astype(np.int64)


def masks_from_keeps(cfg: SlotConfig, case: str, keeps: list[np.ndarray], area: np.ndarray) -> list[np.ndarray]:
    """One kept bool mask per bag for one OOF image. keeps / area: that image's rows (`rows_of_image`). Asserts the
    painted candidates are those rows (same count, same areas), so the masks are exactly what was scored."""
    cl, a = candidate_labels(cfg, case)
    assert np.array_equal(a, area), f"candidate areas != OOF rows ({case})"
    masks = []
    for k in keeps:
        lut = np.zeros(len(a) + 1, bool)
        lut[1:] = k
        masks.append(lut[cl])
    return masks


def kept_masks(cfg: SlotConfig, sel: dict, i: int) -> list[np.ndarray]:
    """`masks_from_keeps` for image i of a `build` result."""
    r = rows_of_image(sel, i)
    return masks_from_keeps(cfg, sel["cases"][i], [k[r] for k in sel["keeps"]], sel["area"][r])
