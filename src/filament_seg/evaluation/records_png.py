"""HISTORICAL per-annotator-record annotation PNGs (`FILAMENT_EXTRACTED/training/<rec>/annotation/*.png`).

Read only under `--records png`, to replicate the reported annotator-agreement and never-seen values, which were
measured on this tree. The default source is `records_coco` (the competition COCO). The competition
ships NO per-record PNGs; this tree is a PNG export of the same records, half a pixel off the pycocotools
rasterisation of the COCO, and a reviewer cannot obtain it.

Layout (one directory per annotator RECORD, 1,154 records over the 707 training observations)::

    training/<idx>-<batch:4><slot:2>-<image stem>/annotation/annotation_class_{1_Left,2_Right,3_Unidentifiable}.png

A class file is absent when the annotator drew nothing of that class. The two rasterisations differ along boundaries
(tens to hundreds of pixels per image), so a number measured on one does not reproduce on the other. Record identity
with the COCO: `records_coco.png_record_key` (checked 1:1 on names and contents, `tests/eval_records/
records_coco_check.py mapping`). No MAGFiLO release file is involved. `union_mask` reproduces the union-label tree on which the reported values were measured, value for value
on 707 / 707 (`tests/eval_records/records_png_union_record.json`).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
from PIL import Image

from filament_seg import filament_paths as fp
from filament_seg.evaluation import host_rows as HR

H = W = 2048
RECORD_RE = re.compile(r"^(\d+)-(\d{4})(\d{2})-(.+)$")     # <idx>-<batch><slot>-<image stem>
CLASS_FILES = ("annotation_class_1_Left.png", "annotation_class_2_Right.png", "annotation_class_3_Unidentifiable.png")


def train_root() -> Path:
    return fp.EXTRACTED / "training"


def records_by_stem(root: Path | None = None) -> dict:
    """lower-case image stem -> [record directory, ...] in sorted directory order. Unparsed names are an error."""
    root = fp.require(Path(root or train_root()), "historical per-record PNG tree (only for --records png)",
                      "it is not in the competition download; the default --records coco needs only the COCO")
    out: dict = {}
    bad = []
    for d in sorted(os.listdir(root)):
        m = RECORD_RE.match(d)
        if not m:
            bad.append(d)
            continue
        out.setdefault(m.group(4).lower(), []).append(root / d)
    if bad:
        raise ValueError(f"{len(bad)} record directories do not parse, e.g. {bad[:3]}")
    return out


def records_by_case(train_case_map, root: Path | None = None) -> dict:
    """case (img_NNNN) -> [record directory, ...]; every case of the mapping must have >= 1 record."""
    stems = HR.case_stems(train_case_map)
    by = records_by_stem(root)
    missing = [c for c, s in stems.items() if s not in by]
    if missing:
        raise ValueError(f"{len(missing)} cases have no annotation record, e.g. {missing[:3]}")
    return {c: by[s] for c, s in stems.items()}


def load_mask(path) -> np.ndarray:
    """A mask PNG as 2-D bool (annotation PNGs are RGB: channel 0 > 0)."""
    a = np.asarray(Image.open(path))
    return (a[..., 0] if a.ndim == 3 else a) > 0


def record_mask(rec_dir: Path) -> np.ndarray:
    """OR of one record's class masks (all-False when the record has no class file)."""
    acc = np.zeros((H, W), bool)
    for cf in CLASS_FILES:
        p = Path(rec_dir) / "annotation" / cf
        if p.is_file():
            m = load_mask(p)
            if m.shape != (H, W):
                raise ValueError(f"{p}: shape {m.shape}")
            acc |= m
    return acc


def union_mask(rec_dirs) -> np.ndarray:
    """The union label of an observation: OR of every class mask of every record (the union labels)."""
    acc = np.zeros((H, W), bool)
    for rd in rec_dirs:
        acc |= record_mask(rd)
    return acc
