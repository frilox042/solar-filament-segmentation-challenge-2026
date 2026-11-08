"""Per-annotator-record masks rasterised from the competition training COCO (`FILAMENT_KAGGLE_COCO`), the DEFAULT.

The competition ships ONE label file, `MAGFiLO_1.0_Annotations_kaggle2026_train.json`: 1,154 COCO images, one per
annotator RECORD, over the 707 training observations, 8,199 polygon annotations. A COCO image id is
`<batch:4><slot:2>-<image stem>` (e.g. `040301-20140609195854Bh`); every record has >= 1 annotation.

RASTERISATION  a record's mask is the OR, over its annotations, of pycocotools `decode(merge(frPyObjects(seg)))` of
    each annotation (`host_rows.decode`), on the 2048 x 2048 frame. The OR over every record of an observation equals
    the shipped nnU-Net training label (Dataset005 `labelsTr`) on 707 / 707 images, bit for bit
    (`tests/test_records_coco.py`, record `tests/eval_records/records_coco_union_record.json`).
ORDER  records of an observation in sorted COCO-id order (as `host_rows.records_by_case`). The PNG source orders them by
    directory name, which differs on 200 of 707 observations; no statistic that reads records depends on the order
    (all ordered record pairs, a mean over records, an OR).
HISTORICAL SOURCE  `records_png` reads `extracted/training/<idx>-<batch><slot>-<stem>/annotation/*.png`, a PNG
    export of the same records, half a pixel off this rasterisation. A reviewer cannot obtain it; it is
    kept only so `--records png` stays a replication check of the reported numbers. Record identity between the two
    is `png_record_key` (case-insensitive `<batch><slot>-<stem>`), checked 1:1 on names and on contents by
    `tests/eval_records/records_mapping.py`.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import numpy as np

from filament_seg import filament_paths as fp
from filament_seg.evaluation import host_rows as HR

H = W = 2048
PNG_DIR_RE = re.compile(r"^\d+-(\d{6}-.+)$")      # <idx>-<batch:4><slot:2>-<image stem>


class Record(NamedTuple):
    """One annotator record: its COCO image id and the `segmentation` of each of its annotations."""
    id: str
    segs: tuple


def coco_path() -> Path:
    return fp.KAGGLE_COCO


@lru_cache(maxsize=2)
def _records_by_stem(path: str) -> dict:
    """lower-case image stem -> [Record, ...] sorted by record id. A record without an annotation is an error."""
    coco = json.loads(Path(path).read_text())
    anns: dict = {}
    for an in coco["annotations"]:
        anns.setdefault(an["image_id"], []).append(an["segmentation"])
    missing = [im["id"] for im in coco["images"] if im["id"] not in anns]
    if missing:
        raise ValueError(f"{len(missing)} COCO records have no annotation, e.g. {missing[:3]}")
    out: dict = {}
    for im in sorted(coco["images"], key=lambda im: im["id"]):
        out.setdefault(im["id"].split("-", 1)[1].lower(), []).append(Record(im["id"], tuple(anns[im["id"]])))
    return out


def records_by_case(train_case_map, path=None) -> dict:
    """case (img_NNNN) -> [Record, ...]; every case of the mapping must have >= 1 record."""
    stems = HR.case_stems(train_case_map)
    by = _records_by_stem(str(path or coco_path()))
    missing = [c for c, s in stems.items() if s not in by]
    if missing:
        raise ValueError(f"{len(missing)} cases have no annotation record, e.g. {missing[:3]}")
    return {c: by[s] for c, s in stems.items()}


def record_mask(rec: Record) -> np.ndarray:
    """OR of one record's annotations, each rasterised by pycocotools (2-D bool, 2048 x 2048)."""
    acc = np.zeros(H * W, bool)
    for seg in rec.segs:
        acc[HR.decode(seg)] = True
    return acc.reshape(H, W)


def union_mask(recs) -> np.ndarray:
    """The union label of an observation: OR of every record's mask (= Dataset005 labelsTr > 0)."""
    acc = np.zeros((H, W), bool)
    for r in recs:
        acc |= record_mask(r)
    return acc


def coco_record_key(record_id: str) -> str:
    """The case-insensitive identity of a record: `<batch><slot>-<stem>` in lower case."""
    return record_id.lower()


def png_record_key(dirname: str) -> str:
    """The same identity from a PNG record directory `<idx>-<batch><slot>-<stem>` (drops the export index)."""
    m = PNG_DIR_RE.match(dirname)
    if not m:
        raise ValueError(f"not a PNG record directory name: {dirname!r}")
    return m.group(1).lower()
