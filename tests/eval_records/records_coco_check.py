#!/usr/bin/env python3
"""Evidence for the COCO per-record source (`filament_seg.evaluation.records_coco`), written as JSON records.

    python tests/eval_records/records_coco_check.py union   OUT.json [--workers 4]
    python tests/eval_records/records_coco_check.py mapping OUT.json [--workers 4]

union    the OR of the COCO records of each observation equals the shipped nnU-Net training label
         (`Dataset005_FilamentFull/labelsTr/<case>.png` > 0), exactly, on every case of the training mapping.
         NEGATIVE CONTROL: the same check with ONE record dropped from the COCO source (the first record of the first
         multi-record observation, by sorted case) must break the identity on exactly that case.
mapping  the record <-> annotator-slot identity between the COCO and the historical PNG tree
         (`extracted/training/<idx>-<batch><slot>-<stem>`): 1:1 on names (case-insensitive `<batch><slot>-<stem>`),
         the class files present equal the COCO categories drawn, and on CONTENTS each PNG record overlaps its own
         COCO record more (IoU) than any sibling record of the observation.
Needs: the competition COCO (FILAMENT_KAGGLE_COCO), the case mapping (WORK), Dataset005 labelsTr (nnUNet_raw; union
only), the PNG tree (FILAMENT_EXTRACTED; mapping only). CPU only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from filament_seg import config as C  # noqa: E402
from filament_seg import filament_paths as fp  # noqa: E402
from filament_seg.evaluation import records_coco as RC  # noqa: E402
from filament_seg.evaluation import records_png as RP  # noqa: E402


def labels_dir() -> Path:
    return fp.DATASET_RAW[5] / "labelsTr"


def _union_one(item):
    case, recs, label = item
    u = RC.union_mask(recs)
    lab = np.asarray(Image.open(label)) > 0
    return case, bool(np.array_equal(u, lab)), int((u ^ lab).sum()), int(lab.sum())


def union_identity(recs_by_case: dict, labels: Path, workers: int = 4) -> dict:
    """{case: (equal, n_xor, label_fg)} for every case."""
    items = [(c, r, labels / f"{c}.png") for c, r in sorted(recs_by_case.items())]
    with ProcessPoolExecutor(workers) as ex:
        return {c: (e, x, f) for c, e, x, f in ex.map(_union_one, items, chunksize=8)}


def negative_control_drop(recs_by_case: dict) -> tuple[str, str, dict]:
    """(case, dropped record id, the source with that one record removed)."""
    case = next(c for c in sorted(recs_by_case) if len(recs_by_case[c]) >= 2)
    drop = recs_by_case[case][0].id
    return case, drop, {c: [r for r in rs if r.id != drop] for c, rs in recs_by_case.items()}


def cmd_union(out: Path, workers: int) -> int:
    cmap = C.train_case_map()
    recs = RC.records_by_case(cmap)
    lab = labels_dir()
    res = union_identity(recs, lab, workers)
    nc_case, nc_drop, nc_recs = negative_control_drop(recs)
    nc = union_identity(nc_recs, lab, workers)
    n_eq = sum(e for e, _, _ in res.values())
    broken = sorted(c for c, (e, _, _) in nc.items() if not e)
    rec = {"what": "OR of the COCO records per observation == Dataset005 labelsTr > 0",
           "coco": RC.coco_path().name, "labels": "/".join(lab.parts[-2:]), "n_cases": len(res),
           "n_records": sum(len(r) for r in recs.values()), "n_equal": n_eq,
           "max_xor": max(x for _, x, _ in res.values()),
           "negative_control": {"dropped_record": nc_drop, "case": nc_case, "n_equal": len(nc) - len(broken),
                                "broken_cases": broken, "xor_px": nc[nc_case][1],
                                "pass": broken == [nc_case]},
           }
    rec["pass"] = bool(n_eq == len(res) == 707 and rec["negative_control"]["pass"])
    out.write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec))
    return 0 if rec["pass"] else 1


def _mapping_one(item):
    case, coco_recs, png_dirs = item
    cm = [RC.record_mask(r) for r in coco_recs]
    ck = [RC.coco_record_key(r.id) for r in coco_recs]
    rows = []
    for d in png_dirs:
        pm = RP.record_mask(d)
        iou = [float((pm & m).sum() / max((pm | m).sum(), 1)) for m in cm]
        key = RC.png_record_key(Path(d).name)
        own = ck.index(key)
        sib = [x for j, x in enumerate(iou) if j != own]
        rows.append({"png": Path(d).name, "own_iou": iou[own], "best_sibling_iou": max(sib) if sib else None,
                     "own_is_best": bool(int(np.argmax(iou)) == own and (not sib or iou[own] > max(sib)))})
    return case, rows


def cmd_mapping(out: Path, workers: int) -> int:
    cmap = C.train_case_map()
    coco = RC.records_by_case(cmap)
    png = RP.records_by_case(cmap)
    # names: 1:1 on the case-insensitive key, per observation and overall
    ck = sorted(RC.coco_record_key(r.id) for rs in coco.values() for r in rs)
    pk = sorted(RC.png_record_key(Path(d).name) for ds in png.values() for d in ds)
    per_case_names = all(sorted(RC.coco_record_key(r.id) for r in coco[c])
                         == sorted(RC.png_record_key(Path(d).name) for d in png[c]) for c in coco)
    # class files present == COCO categories drawn
    cats: dict = {}
    for an in json.loads(RC.coco_path().read_text())["annotations"]:
        cats.setdefault(RC.coco_record_key(an["image_id"]), set()).add(an["category_id"])
    cls_bad = [Path(d).name for ds in png.values() for d in ds
               if {i + 1 for i, f in enumerate(RP.CLASS_FILES) if (Path(d) / "annotation" / f).is_file()}
               != cats[RC.png_record_key(Path(d).name)]]
    items = [(c, coco[c], png[c]) for c in sorted(coco)]
    with ProcessPoolExecutor(workers) as ex:
        rows = dict(ex.map(_mapping_one, items, chunksize=4))
    flat = [r for rs in rows.values() for r in rs]
    multi = [r for c, rs in rows.items() if len(rs) >= 2 for r in rs]
    rec = {"what": "COCO record <-> PNG record (annotator slot) identity", "n_cases": len(coco),
           "n_coco_records": len(ck), "n_png_records": len(pk), "names_equal": ck == pk,
           "names_unique": len(set(ck)) == len(ck), "per_case_names_equal": per_case_names,
           "class_files_vs_categories_mismatch": cls_bad,
           "content": {"n_records": len(flat), "n_multi_record_obs": sum(len(rs) >= 2 for rs in rows.values()),
                       "n_records_in_multi": len(multi), "own_is_best_all": all(r["own_is_best"] for r in flat),
                       "not_own_best": [r["png"] for r in flat if not r["own_is_best"]],
                       "own_iou_min": min(r["own_iou"] for r in flat),
                       "own_iou_median": float(np.median([r["own_iou"] for r in flat])),
                       "best_sibling_iou_max": max(r["best_sibling_iou"] for r in multi),
                       "min_margin_own_minus_sibling": min(r["own_iou"] - r["best_sibling_iou"] for r in multi)},
           "png_order_differs_from_coco_order_cases": sum(
               [RC.png_record_key(Path(d).name) for d in png[c]] != [RC.coco_record_key(r.id) for r in coco[c]]
               for c in coco)}
    rec["pass"] = bool(rec["names_equal"] and rec["names_unique"] and per_case_names and not cls_bad
                       and rec["content"]["own_is_best_all"] and len(ck) == 1154)
    out.write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec))
    return 0 if rec["pass"] else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("union", "mapping"))
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    a = ap.parse_args()
    return (cmd_union if a.what == "union" else cmd_mapping)(Path(a.out), a.workers)


if __name__ == "__main__":
    sys.exit(main())
