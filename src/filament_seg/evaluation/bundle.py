"""The out-of-fold evaluation bundle (`eval_bundle/`): what `scripts/eval_*.py` need, without the 2048² maps.

The offline evaluation reads four model-side inputs, all out of fold (OOF) on the 707 training observations:
the base network's probability maps, its argmax binaries (the D007 >= 400 px reference arm), the selector's OOF rows
(features of every candidate, built from the maps, the two member masks and channel 0), and a per-frame limb fit.
Regenerating them needs the full retrain (`docs/TRAINING.md`). The bundle ships them in a compact form:

  oof_rows_nnHALO.npz     the selector's OOF rows: X (22 features per candidate), area, and the row accounting
                        `Ref` scores (image index, per-record match numbers coco_v / mp / M / miss, per-image record and
                        annotation counts coco_k / coco_G). Numbers only, no geometry. `oof_selection.build` reads it
                        in place of `selector_fit.build_oof_table` (which needs the maps and member masks).
  oof_levels/<case>.png the base map reduced to the threshold LEVELS the evaluation reads (uint8 codes, `LEVELS`):
                        p >= 0.97 (candidates), >= 0.92 (morphometry's negative control), < 0.3 (barb pixels),
                        < 0.1 and > 0.5 (never-seen classes). `stage` expands it to a uint16 map whose every pixel holds
                        the smallest uint16 value of its level, so every threshold the evaluation applies gives the same
                        answer as on the original map, under both comparisons the code uses (a float32 array against a
                        Python float, and a Python float of a float32 value). Continuous values are NOT kept: the
                        `p_max` column of never_seen_rows.npz holds level values, not probabilities.
  d007_oof_bin/<case>.png  the D007 OOF argmax binaries (our own prediction masks), unchanged.
  diskfit.json          per-frame limb fit (disk centre and radius, image-derived): the never-seen r/R feature.
  bundle.json, MANIFEST.sha256   contents, provenance, and the sha256 of every file (checked by `stage`).

Not in the bundle, by the competition rules (data for non-commercial use, not redistributed): images and
annotations. `stage` reads them from YOUR download (`$FILAMENT_KAGGLE_ROOT`): the training COCO (every per-record
reference is rasterised from it) and the 707 training JPEGs, from which it writes channel 0 (`prep.channels`, the
min-max uint16 of the raw frame) for the never-seen contrast and clutter features. No annotation geometry is shipped:
the rows hold IoU sums and match counts only.

    python -m filament_seg.evaluation.bundle stage --bundle eval_bundle --work WORK     # reviewers
    python -m filament_seg.evaluation.bundle export --out eval_bundle                   # how it was made (full maps)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from filament_seg import filament_paths as fp

ROWS = "oof_rows_{slot}.npz"
LEVELS_DIR, BIN_DIR, DISKFIT, INDEX, MANIFEST = "oof_levels", "d007_oof_bin", "diskfit.json", "bundle.json", \
    "MANIFEST.sha256"
# Every probability threshold the evaluation applies (module: what it decides). Both comparison semantics are kept.
LEVELS = ((0.1, ">="), (0.3, ">="), (0.5, ">"), (0.92, ">="), (0.97, ">="))
LEVEL_USES = {"0.1": "never_seen: class D = p_max < 0.1", "0.3": "morphometry: barb pixel outside and p < 0.3",
              "0.5": "never_seen: class B = p_max > 0.5", "0.92": "morphometry: negative control base p >= 0.92",
              "0.97": "candidates: base mask p >= 0.97 (configs/nnHALO.json core_t)"}
# Written into the staged map directory: the selector fit refuses a level map (its features need real probabilities).
MARKER = "LEVEL_MAPS_FROM_EVAL_BUNDLE.json"
# WORK paths the staged bundle fills (configs/eval/inputs.json, config.py)
W_PROB, W_BIN = "predictions/vessel_oof_probs", "predictions/vessel_oof_full/by_image"
W_DISKFIT, W_ROWS = "analysis/datastudy_diskfit.json", "analysis/" + ROWS
W_CASEMAPS = ("nnunet_env/nnUNet_raw/Dataset005_FilamentFull/case_mapping_full.json",
              "nnunet_env/nnUNet_raw/Dataset007_FilamentVessel/case_mapping_full.json")
W_CH0 = "nnunet_env/nnUNet_raw/Dataset007_FilamentVessel/imagesTr"


# ---------------------------------------------------------------------------------------------- levels
def _signature(v: np.ndarray) -> np.ndarray:
    """(65536, 2 x len(LEVELS)) bool: every threshold on every uint16 value, as an array op on the float32 map (numpy
    casts the Python float to float32) and as a Python-float scalar op (float64 of the float32 value)."""
    p32 = v.astype(np.float32) / np.float32(65535.0)          # = maps.load_prob_png
    p64 = p32.astype(np.float64)
    cols = []
    for t, op in LEVELS:
        for p in (p32, p64):
            cols.append(p >= t if op == ">=" else p > t)
    return np.stack(cols, 1)


def level_table() -> tuple[np.ndarray, np.ndarray]:
    """(code of each uint16 value, uint8; the smallest uint16 value of each code). Codes number the distinct
    signatures in increasing value order; the signature is monotone in the value, so each code is an interval."""
    v = np.arange(65536, dtype=np.uint16)
    sig = _signature(v)
    change = np.r_[True, np.any(sig[1:] != sig[:-1], 1)]
    code = (np.cumsum(change) - 1).astype(np.uint8)
    rep = v[change]
    assert len(np.unique([s.tobytes() for s in sig[change]])) == len(rep), "a signature repeats: not monotone"
    return code, rep


def to_levels(u16: np.ndarray) -> np.ndarray:
    return level_table()[0][u16]


def from_levels(codes: np.ndarray) -> np.ndarray:
    return level_table()[1][codes]


# ---------------------------------------------------------------------------------------------- integrity
def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_manifest(root: Path) -> None:
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name != MANIFEST)
    (root / MANIFEST).write_text("".join(f"{sha256(p)}  {p.relative_to(root).as_posix()}\n" for p in files))


def check_manifest(root: Path) -> int:
    """Every file listed matches its sha256 and no unlisted file is present; returns the number of files."""
    root = Path(root)
    listed = {}
    for line in (root / MANIFEST).read_text().splitlines():
        h, rel = line.split(None, 1)
        listed[rel] = h
    present = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.name != MANIFEST}
    if present != set(listed):
        raise ValueError(f"bundle files differ from {MANIFEST}: missing {sorted(set(listed) - present)[:5]}, "
                         f"unlisted {sorted(present - set(listed))[:5]}")
    bad = [rel for rel, h in listed.items() if sha256(root / rel) != h]
    if bad:
        raise ValueError(f"sha256 mismatch in the bundle: {bad[:5]}")
    return len(listed)


# ---------------------------------------------------------------------------------------------- reading the rows
def rows_path(slot: str) -> Path:
    """Where a staged bundle puts the selector's OOF rows (WORK/analysis/oof_rows_<slot>.npz)."""
    return fp.WORK / W_ROWS.format(slot=slot)


def load_rows(path) -> tuple[dict, np.ndarray, np.ndarray]:
    """(Zc, X, area) exactly as `selector_fit.build_oof_table` returns them."""
    z = np.load(path, allow_pickle=False)
    Zc = {k[2:]: z[k] for k in z.files if k.startswith("Z_")}
    return Zc, z["X"], z["area"]


# ---------------------------------------------------------------------------------------------- export
def export(out: Path, slot: str = "nnHALO", workers: int = 3) -> dict:
    """Write the bundle from a WORK holding the full OOF maps (the research run root; not needed by reviewers)."""
    from concurrent.futures import ProcessPoolExecutor

    from filament_seg import config as C
    from filament_seg.selector_fit import build_oof_table
    cfg = C.load(slot)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    Zc, X, area = build_oof_table(cfg)
    np.savez_compressed(out / ROWS.format(slot=slot), X=X, area=area,
                        **{f"Z_{k}": np.asarray(v) for k, v in Zc.items()})
    cases = [str(c) for c in Zc["cases"]]
    (out / LEVELS_DIR).mkdir(exist_ok=True)
    (out / BIN_DIR).mkdir(exist_ok=True)
    with ProcessPoolExecutor(workers) as ex:
        list(ex.map(_export_one, [(str(cfg.oof_prob_dir), str(fp.WORK / W_BIN), str(out), c) for c in cases],
                    chunksize=8))
    fits = json.loads((fp.WORK / W_DISKFIT).read_text())["frames"]
    keep = ("stem", "fit_x", "fit_y", "fit_R")
    (out / DISKFIT).write_text(json.dumps({"frames": [{k: f[k] for k in keep} for f in fits]}, indent=0))
    code, rep = level_table()
    index = {"what": __doc__.split("\n")[0], "slot": slot, "n_cases": len(cases), "n_rows": int(len(area)),
             "n_features": int(X.shape[1]),
             "levels": {"thresholds": [list(t) for t in LEVELS], "uses": LEVEL_USES,
                        "uint16_of_code": rep.tolist()},
             "sources": {"oof_rows": "selector_fit.build_oof_table on the OOF maps (configs/nnHALO.json cpu_stage)",
                         "oof_levels": cfg.oof_prob_dir.relative_to(fp.WORK).as_posix(), "d007_oof_bin": W_BIN,
                         "diskfit": W_DISKFIT + " (frames: stem, fit_x, fit_y, fit_R)"}}
    (out / INDEX).write_text(json.dumps(index, indent=1))
    write_manifest(out)
    return index


def _export_one(args) -> None:
    prob_dir, bin_dir, out, case = args
    u16 = np.asarray(Image.open(f"{prob_dir}/{case}.png"))
    assert u16.dtype == np.uint16 and u16.shape == (2048, 2048), (case, u16.dtype, u16.shape)
    Image.fromarray(to_levels(u16)).save(f"{out}/{LEVELS_DIR}/{case}.png", optimize=True)
    b = np.asarray(Image.open(f"{bin_dir}/{case}.png"))
    Image.fromarray(((b[..., 0] if b.ndim == 3 else b) > 0)).save(f"{out}/{BIN_DIR}/{case}.png", optimize=True)


# ---------------------------------------------------------------------------------------------- stage
def stage(bundle: Path, work: Path | None = None, kaggle_root: Path | None = None, workers: int = 3) -> dict:
    """Fill WORK with the evaluation's inputs from the bundle (after checking its manifest) and channel 0 of the 707
    training JPEGs of your download. Existing WORK files are never overwritten (re-runs resume)."""
    from concurrent.futures import ProcessPoolExecutor

    from filament_seg.prep import prepare
    bundle, work = Path(bundle), Path(work or fp.WORK)
    n_files = check_manifest(bundle)
    index = json.loads((bundle / INDEX).read_text())
    slot = index["slot"]
    _, rep = level_table()
    if rep.tolist() != index["levels"]["uint16_of_code"]:
        raise ValueError("this code's level table differs from the bundle's")
    cm = json.loads((fp.CONFIGS / "case_mapping_full.json").read_text())
    cases = sorted(cm["train"])
    root = Path(kaggle_root or fp.KAGGLE_ROOT)
    jpegs = {c: prepare.kaggle_image(root, "train", cm["train"][c]) for c in cases}
    miss = [str(p) for p in jpegs.values() if not p.exists()]
    if miss:
        raise FileNotFoundError(f"{len(miss)} training JPEGs not found under FILAMENT_KAGGLE_ROOT={root}, "
                                f"e.g. {miss[0]}")
    if not fp.KAGGLE_COCO.exists():
        raise FileNotFoundError(f"competition training COCO not found: {fp.KAGGLE_COCO} (set FILAMENT_KAGGLE_ROOT)")
    for rel in W_CASEMAPS:
        p = work / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            p.write_text(json.dumps(cm, indent=2))           # = prep.prepare.write_bookkeeping
    for rel, src in ((W_DISKFIT, DISKFIT), (W_ROWS.format(slot=slot), ROWS.format(slot=slot))):
        (work / rel).parent.mkdir(parents=True, exist_ok=True)
        if not (work / rel).exists():
            shutil.copyfile(bundle / src, work / rel)
    for d in (W_PROB, W_BIN, W_CH0):
        (work / d).mkdir(parents=True, exist_ok=True)
    (work / W_PROB / MARKER).write_text(json.dumps(
        {"what": "uint16 maps expanded from the evaluation bundle's threshold levels, NOT probabilities",
         "levels": index["levels"], "bundle_manifest_sha256": sha256(bundle / MANIFEST)}, indent=1))
    items = [(str(bundle), str(work), c, str(jpegs[c])) for c in cases]
    with ProcessPoolExecutor(workers) as ex:
        list(ex.map(_stage_one, items, chunksize=8))
    return {"work": str(work), "bundle_files": n_files, "cases": len(cases), "slot": slot,
            "rows": W_ROWS.format(slot=slot), "kaggle_root": "FILAMENT_KAGGLE_ROOT"}


def _stage_one(args) -> None:
    from filament_seg.prep import prepare
    from filament_seg.prep.channels import to_u16
    bundle, work, case, jpeg = args
    for src, dst, fn in ((f"{bundle}/{LEVELS_DIR}/{case}.png", f"{work}/{W_PROB}/{case}.png", from_levels),
                         (f"{bundle}/{BIN_DIR}/{case}.png", f"{work}/{W_BIN}/{case}.png",
                          lambda a: (a > 0).astype(np.uint8))):
        if not os.path.exists(dst):
            _save(fn(np.asarray(Image.open(src))), dst)
    dst = f"{work}/{W_CH0}/{case}_0000.png"
    if not os.path.exists(dst):
        _save(to_u16(prepare.raw_u8(Path(jpeg)).astype(np.float32)), dst)     # = four_channels(raw)[0]


def _save(a: np.ndarray, dst: str) -> None:
    tmp = dst + ".part"
    Image.fromarray(a).save(tmp, format="PNG")
    os.replace(tmp, dst)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage")
    s.add_argument("--bundle", type=Path, default=fp.REPO / "eval_bundle")
    s.add_argument("--work", type=Path)
    s.add_argument("--workers", type=int, default=3)
    e = sub.add_parser("export")
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--workers", type=int, default=3)
    c = sub.add_parser("check")
    c.add_argument("--bundle", type=Path, default=fp.REPO / "eval_bundle")
    a = ap.parse_args(argv)
    if a.cmd == "stage":
        print(json.dumps(stage(a.bundle, a.work, workers=a.workers)))
    elif a.cmd == "export":
        print(json.dumps(export(a.out, workers=a.workers))[:400])
    else:
        print(json.dumps({"files_verified": check_manifest(a.bundle)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
