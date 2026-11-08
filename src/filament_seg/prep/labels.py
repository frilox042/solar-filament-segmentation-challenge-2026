"""Training labels of the nnU-Net members, built from the competition download (the full-retrain tier).

    python scripts/build_labels.py [--datasets D005 D007 D012 D036] [--limit N] [--workers 4]
                                   [--kaggle-root DIR] [--coco FILE]

SOURCE  the competition's own training annotation file, `MAGFiLO_1.0_Annotations_kaggle2026_train.json`
    (`$FILAMENT_KAGGLE_COCO`; 1,154 annotator records over the 707 training images, 8,199 polygons), and the
    `train_images/` listing of the download (`$FILAMENT_KAGGLE_ROOT/MAGFiLO_1.0_Kaggle_2026/train/train_images`).
    Nothing else is read; in particular not the MAGFiLO release JSON.
CASES  `img_XXXX` numbers the training JPEGs in sorted file-name order. The enumeration is derived from the listing
    and asserted equal to `configs/case_mapping_full.json["train"]` (the enumeration the shipped members were trained on), so a different
    download fails loudly instead of renumbering the cases.
UNION (Dataset005 / 007 / 012)  `labelsTr/<case>.png`, uint8 mode L, 0 background / 1 filament: the OR over every
    record of the image of pycocotools `decode(merge(frPyObjects(polygon)))` of each annotation
    (`filament_seg.evaluation.records_coco.union_mask`, the single rasteriser of this repository).
BARB CLASS (Dataset036)  the same file with a third class, 0 / 1 filament / 2 barb: `barb_label(union)`, with
    `spine_and_barbs` for the skeleton and geodesic-diameter path. Barbs are
    geometric: every skeleton pixel off the geodesic-diameter path of an object of >= 200 px, dilated to the object's
    own median half-width and clipped to the object.
IDENTITY  on the 707 training images both outputs equal the labels the shipped members were trained on, value for
    value (check `tests/eval_records/build_labels_check.py`, record `tests/eval_records/build_labels_record.json`,
    `tests/test_labels.py`).

Resume-safe: a label that already exists is not rebuilt (`--verify-against` compares instead of writing).
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage, sparse
from scipy.sparse.csgraph import dijkstra
from skimage.morphology import skeletonize

from filament_seg import filament_paths as fp
from filament_seg.evaluation import records_coco as RC

H = W = 2048
SQ2 = float(np.sqrt(2.0))
MIN_OBJ = 200          # below this an object has no meaningful spine/barb split

DATASETS = {"D005": "Dataset005_FilamentFull", "D007": "Dataset007_FilamentVessel",
            "D012": "Dataset012_FilamentRaw707", "D036": "Dataset036_LimbDarkBarbClass"}
KIND = {"D005": "union", "D007": "union", "D012": "union", "D036": "barb"}
TRAIN_IMAGES = Path("MAGFiLO_1.0_Kaggle_2026") / "train" / "train_images"


# ------------------------------------------------------------------------------------------------ case enumeration
def enumerate_cases(kaggle_root: Path | None = None) -> dict:
    """{img_XXXX: file name} from the sorted `train_images/` listing; must equal the enumeration the shipped members were trained on."""
    d = Path(kaggle_root or fp.KAGGLE_ROOT) / TRAIN_IMAGES
    fp.require(d, "competition training images", "unpack filament-segmentation-2026.zip into $FILAMENT_KAGGLE_ROOT")
    files = sorted(f for f in os.listdir(d) if f.lower().endswith((".jpeg", ".jpg")))
    cases = {f"img_{i:04d}": f for i, f in enumerate(files)}
    ref = json.loads((fp.CONFIGS / "case_mapping_full.json").read_text())["train"]
    if cases != ref:
        diff = sorted(k for k in set(cases) | set(ref) if cases.get(k) != ref.get(k))
        raise ValueError(f"train_images listing ({len(cases)} files) does not reproduce configs/case_mapping_full.json"
                         f" ({len(ref)} cases); first differing cases {diff[:3]}")
    for f in files:                                     # header only: the labels assume the 2048 x 2048 frame
        with Image.open(d / f) as im:
            if im.size != (W, H):
                raise ValueError(f"{f}: image size {im.size}, expected {(W, H)}")
    return cases


# ------------------------------------------------------------------------------------- barb class (verbatim port)
def spine_and_barbs(skel):
    """Split a skeleton into its geodesic-diameter path (spine) and everything else (barbs).

    Returns (spine_mask, barb_mask). Exact for tree skeletons; with loops the diameter is
    approximate. Edge order, Dijkstra from index 0 and argmax ties are fixed so the labels are reproducible.
    """
    ys, xs = np.nonzero(skel)
    n = len(ys)
    if n == 0:
        return np.zeros_like(skel), np.zeros_like(skel)
    if n == 1:
        return skel.copy(), np.zeros_like(skel)

    idx = np.full(skel.shape, -1, np.int64)
    idx[ys, xs] = np.arange(n)
    P = np.pad(idx, 1, constant_values=-1)
    core = P[1:skel.shape[0] + 1, 1:skel.shape[1] + 1]
    r, c, w = [], [], []
    for dy, dx, wt in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, SQ2), (1, -1, SQ2)):
        nb = P[1 + dy:1 + dy + skel.shape[0], 1 + dx:1 + dx + skel.shape[1]]
        m = (core >= 0) & (nb >= 0)
        if m.any():
            r.append(core[m])
            c.append(nb[m])
            w.append(np.full(int(m.sum()), wt))
    if not r:
        return skel.copy(), np.zeros_like(skel)
    r = np.concatenate(r)
    c = np.concatenate(c)
    w = np.concatenate(w)
    g = sparse.coo_matrix((np.concatenate([w, w]),
                           (np.concatenate([r, c]), np.concatenate([c, r]))),
                          shape=(n, n)).tocsr()

    d0 = dijkstra(g, indices=0)
    u = int(np.argmax(np.where(np.isfinite(d0), d0, -1)))
    du, pred = dijkstra(g, indices=u, return_predecessors=True)
    v = int(np.argmax(np.where(np.isfinite(du), du, -1)))

    path = []
    k = v
    while k != -9999 and k >= 0:
        path.append(k)
        if k == u:
            break
        k = int(pred[k])
    spine = np.zeros_like(skel)
    if path:
        spine[ys[path], xs[path]] = True
    barbs = skel & ~spine
    return spine, barbs


def barb_label(union):
    """0 background, 1 filament body, 2 barb. See the module docstring.

    The barb region is the barb SKELETON dilated to the object's own local half-width, so the class
    covers the physical barb rather than a one-pixel line.
    """
    out = np.zeros(union.shape, np.uint8)
    lab, n = ndimage.label(union)
    if n == 0:
        return out
    objs = ndimage.find_objects(lab)
    for i in range(1, n + 1):
        sl = objs[i - 1]
        m = (lab[sl] == i)
        if m.sum() < MIN_OBJ:
            out[sl][m] = 1
            continue
        sk = skeletonize(m)
        if sk.sum() < 5:
            out[sl][m] = 1
            continue
        _, barbs = spine_and_barbs(sk)
        sub = np.zeros(m.shape, np.uint8)
        sub[m] = 1
        if barbs.any():
            edt = ndimage.distance_transform_edt(m)
            if edt is None:          # only when an output array is passed; never with these arguments
                raise RuntimeError("distance_transform_edt returned no distances")
            half = max(float(np.median(edt[sk])), 1.0)      # the object's OWN half-width
            grown = ndimage.binary_dilation(barbs, iterations=int(round(half))) & m
            sub[grown] = 2
        out[sl] = np.maximum(out[sl], sub)
    return out


# --------------------------------------------------------------------------------------------------------- build
def labels_for(recs, kinds) -> dict:
    """{"union": uint8 0/1, "barb": uint8 0/1/2} for the requested kinds, from one image's records."""
    u = RC.union_mask(recs)
    out = {}
    if "union" in kinds:
        out["union"] = u.astype(np.uint8)
    if "barb" in kinds:
        out["barb"] = barb_label(u)
    return out


def write_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    Image.fromarray(arr, mode="L").save(tmp, format="PNG")
    tmp.replace(path)


def _build_one(item):
    case, recs, targets = item                     # targets: {dataset key: output path}
    todo = {k: p for k, p in targets.items() if not Path(p).exists()}
    if not todo:
        return case, 0
    labs = labels_for(recs, {KIND[k] for k in todo})
    for k, p in todo.items():
        write_png(Path(p), labs[KIND[k]])
    return case, len(todo)


def _verify_one(item):
    case, recs, refs = item                        # refs: {dataset key: reference label path}
    labs = labels_for(recs, {KIND[k] for k in refs})
    res = {}
    for k, p in refs.items():
        ref = np.asarray(Image.open(p))
        ref = ref[..., 0] if ref.ndim == 3 else ref
        a = labs[KIND[k]]
        res[k] = {"equal": bool(ref.shape == a.shape and np.array_equal(a, ref)), "n_diff": int((a != ref).sum()),
                  "fg": int((ref > 0).sum()), "barb": int((ref == 2).sum())}
    return case, res


def _items(cases, recs_by_case, per_dataset, limit):
    keys = sorted(cases)[: limit or None]
    return [(c, recs_by_case[c], {k: str(f(c)) for k, f in per_dataset.items()}) for c in keys]


def records(coco: Path | None = None, kaggle_root: Path | None = None) -> tuple[dict, dict]:
    """(cases, {case: [Record, ...]}) from the download; the case enumeration is checked first."""
    cases = enumerate_cases(kaggle_root)
    coco = Path(coco or fp.KAGGLE_COCO)
    fp.require(coco, "competition training COCO JSON", "it ships in filament-segmentation-2026.zip (train/)")
    # the listing equals the mapping (checked above), so the mapping file names the same cases
    return cases, RC.records_by_case(fp.CONFIGS / "case_mapping_full.json", coco)


def build(datasets, coco=None, kaggle_root=None, raw=None, limit=0, workers=4) -> dict:
    """Write `<raw>/<Dataset>/labelsTr/<case>.png` for every case; returns {dataset: files written}."""
    raw_dir: Path = Path(raw) if raw is not None else fp.NNUNET_RAW
    cases, recs = records(coco, kaggle_root)
    per = {k: (lambda c, k=k: raw_dir / DATASETS[k] / "labelsTr" / f"{c}.png") for k in datasets}
    items = _items(cases, recs, per, limit)
    n = 0
    with ProcessPoolExecutor(workers) as ex:
        for _, k in ex.map(_build_one, items, chunksize=4):
            n += k
    return {"cases": len(items), "files_written": n, "datasets": [DATASETS[k] for k in datasets], "raw": str(raw_dir)}


def verify(datasets, reference_raw, coco=None, kaggle_root=None, limit=0, workers=4, recs_override=None) -> dict:
    """Rebuild in memory and compare value for value with `<reference_raw>/<Dataset>/labelsTr`. `recs_override`
    replaces the records (negative controls)."""
    ref = Path(reference_raw)
    cases, recs = records(coco, kaggle_root)
    if recs_override is not None:
        recs = recs_override
    per = {k: (lambda c, k=k: ref / DATASETS[k] / "labelsTr" / f"{c}.png") for k in datasets}
    with ProcessPoolExecutor(workers) as ex:
        return dict(ex.map(_verify_one, _items(cases, recs, per, limit), chunksize=4))
