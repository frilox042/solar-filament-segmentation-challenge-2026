"""Inputs from the raw competition download: the members' channel images (and the run root's bookkeeping files).

    python -m filament_seg.prep.prepare --channels [--split train|test|both] [--datasets D005 D007 D012 D036]
                                        [--limit N] [--kaggle-root DIR] [--workers W]
    python -m filament_seg.prep.prepare --check N            # rebuild N images per split, compare with $FILAMENT_WORK

Reads `$FILAMENT_KAGGLE_ROOT/MAGFiLO_1.0_Kaggle_2026/{train/train_images,test/test_images}/<image>.jpeg` (the files of
`filament-segmentation-2026.zip`) and writes, under `$FILAMENT_WORK/nnunet_env/nnUNet_raw/<Dataset>/images{Tr,Ts}/`,
`<case>_<cccc>.png` uint16 channels (`filament_seg.prep.channels`). Case ids come from `configs/case_mapping_full.json`
(img_XXXX for the 707 training images, filamentTs_XXXX for the 180 test images). Also writes each dataset's
`dataset.json` (configs/datasets/) and `case_mapping_full.json`.
Resume-safe: an image whose channels all exist is skipped. Training LABELS are built by `filament_seg.prep.labels`
(`scripts/build_labels.py`).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from filament_seg import filament_paths as fp
from filament_seg.prep.channels import all_channels

SHORT = {"D005": "Dataset005_FilamentFull", "D007": "Dataset007_FilamentVessel", "D012": "Dataset012_FilamentRaw707",
         "D036": "Dataset036_LimbDarkBarbClass"}
SPLIT_DIR = {"train": ("train/train_images", "imagesTr"), "test": ("test/test_images", "imagesTs")}


def case_mapping() -> dict:
    """{"train": {img_XXXX: file}, "test": {filamentTs_XXXX: file}}."""
    return json.loads((fp.CONFIGS / "case_mapping_full.json").read_text())


def kaggle_image(root: Path, split: str, fname: str) -> Path:
    return root / "MAGFiLO_1.0_Kaggle_2026" / SPLIT_DIR[split][0] / fname


def raw_u8(path: Path) -> np.ndarray:
    """The raw 8-bit frame exactly as the shipped datasets stored it (JPEG -> PIL mode L)."""
    return np.asarray(Image.open(path).convert("L"))


def write_bookkeeping(datasets: list) -> None:
    """Each built dataset's dataset.json and case map, and always Dataset005's case map: Dataset005's directory is the
    holder of the TRAINING case map (`config.TRAIN_CASE_MAP`) that the selector fit and the evaluation read."""
    cm = case_mapping()
    holder = fp.NNUNET_RAW / SHORT["D005"]
    holder.mkdir(parents=True, exist_ok=True)
    (holder / "case_mapping_full.json").write_text(json.dumps(cm, indent=2))
    for d in datasets:
        dst = fp.NNUNET_RAW / SHORT[d]
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(fp.CONFIGS / "datasets" / f"{SHORT[d]}.dataset.json", dst / "dataset.json")
        (dst / "case_mapping_full.json").write_text(json.dumps(cm, indent=2))


def _channel_files(datasets: list, sub: str, case: str) -> dict:
    return {d: [fp.NNUNET_RAW / SHORT[d] / sub / f"{case}_{c:04d}.png"
                for c in range(len(json.loads((fp.CONFIGS / "datasets" / f"{SHORT[d]}.dataset.json").read_text())
                                   ["channel_names"]))] for d in datasets}


def _build_one(job: tuple) -> int:
    """Write one case's channels (1 if built, 0 if already complete). An array shared by several datasets (e.g. D036's
    first five channels are D007's) is encoded once and HARD-LINKED for the others: identical bytes, no dangling link
    when a dataset is built on its own."""
    root, split, case, fname, datasets = job
    outs = _channel_files(datasets, SPLIT_DIR[split][1], case)
    if all(p.exists() for ps in outs.values() for p in ps):
        return 0
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):
        chans = all_channels(raw_u8(kaggle_image(root, split, fname)))
    first: dict = {}                                  # id(array) -> the first file written for it
    for d, ps in outs.items():
        arrs = chans[SHORT[d]]
        assert len(arrs) == len(ps), (d, len(arrs), len(ps))
        for p, a in zip(ps, arrs, strict=True):
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".part")
            tmp.unlink(missing_ok=True)
            src = first.get(id(a))
            if src is not None:
                try:
                    os.link(src, tmp)
                except OSError:                       # no hard links here (EXDEV / EPERM): an identical copy
                    shutil.copyfile(src, tmp)
            else:
                Image.fromarray(a).save(tmp, format="PNG")   # uint16 -> mode I;16
                first[id(a)] = p
            tmp.replace(p)
    return 1


def build(split: str, datasets: list, limit: int = 0, root: Path | None = None, workers: int = 1) -> int:
    """Write the channels of every (or the first `limit`) case of a split, `workers` images in parallel (the output is
    byte-identical for any worker count); returns how many images were built."""
    root = Path(root or fp.KAGGLE_ROOT)
    items = sorted(case_mapping()[split].items())[: limit or None]
    jobs = [(root, split, case, fname, datasets) for case, fname in items]
    if workers <= 1:
        return sum(map(_build_one, jobs))
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(workers) as ex:
        return sum(ex.map(_build_one, jobs, chunksize=1))


def check(n: int, root: Path | None = None, reference: Path | None = None) -> dict:
    """Rebuild the first n images of each split IN MEMORY and compare every channel array with the stored ones."""
    root = Path(root or fp.KAGGLE_ROOT)
    ref = Path(reference or fp.NNUNET_RAW)
    out = {}
    for split in ("train", "test"):
        sub = SPLIT_DIR[split][1]
        for case, fname in sorted(case_mapping()[split].items())[:n]:
            chans = all_channels(raw_u8(kaggle_image(root, split, fname)))
            for ds, arrs in chans.items():
                for c, a in enumerate(arrs):
                    stored = np.asarray(Image.open(ref / ds / sub / f"{case}_{c:04d}.png"))
                    out[f"{split}/{ds}/{case}_{c:04d}"] = bool(np.array_equal(a, stored))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channels", action="store_true")
    ap.add_argument("--split", choices=("train", "test", "both"), default="both")
    ap.add_argument("--datasets", nargs="+", default=list(SHORT), choices=list(SHORT))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--kaggle-root", type=Path)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("FILAMENT_CHANNEL_WORKERS", "4")),
                    help="images built in parallel (default 4, or $FILAMENT_CHANNEL_WORKERS); output identical for any value")
    ap.add_argument("--check", type=int, default=0, help="compare N rebuilt images per split with --reference")
    ap.add_argument("--reference", type=Path, help="nnUNet_raw tree to compare with (default: $FILAMENT_WORK's)")
    a = ap.parse_args(argv)
    if a.check:
        r = check(a.check, a.kaggle_root, a.reference)
        print(json.dumps({"n_arrays": len(r), "all_equal": all(r.values()),
                          "unequal": [k for k, v in r.items() if not v]}))
        return 0 if all(r.values()) else 1
    if not a.channels:
        sys.exit("nothing to do: pass --channels or --check N")
    write_bookkeeping(a.datasets)
    for split in (("train", "test") if a.split == "both" else (a.split,)):
        print(f"{split}: built {build(split, a.datasets, a.limit, a.kaggle_root, a.workers)} image(s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
