"""Per-image tables the FLOOR selector's fit needs, built from competition data only (a `prepare_data` step).

The selector's out-of-fold table (`floor_table`) takes three per-image arrays:
  cases    the 707 training case ids (img_XXXX), sorted
  coco_k   number of annotator RECORDS of that image that drew at least one filament (competition COCO image ids)
  coco_G   number of annotated filaments over all records (competition COCO annotations)
They are built from the competition training COCO JSON and the shipped case mapping
(`configs/case_mapping_full.json`); the step does not depend on the channels step.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from filament_seg import filament_paths as fp

TABLE = "analysis/selector_tables.npz"          # relative to WORK


def build() -> dict:
    """{cases, coco_k, coco_G} from the competition COCO + the shipped case mapping.

    The mapping is read from `configs/case_mapping_full.json` (the enumeration of the shipped members; `prep.prepare` copies it
    verbatim into every dataset's `case_mapping_full.json`), so this step needs neither the channels step nor any
    nnU-Net tree: it can run first."""
    full = json.loads((fp.CONFIGS / "case_mapping_full.json").read_text())["train"]
    stem_of = {k: os.path.splitext(v if isinstance(v, str) else v["image"])[0].lower() for k, v in full.items()}
    coco = json.load(open(fp.KAGGLE_COCO))
    recs: dict[str, dict[str, int]] = {}
    for im in coco["images"]:
        recs.setdefault(im["id"].split("-", 1)[1].lower(), {})[im["id"]] = 0
    for an in coco["annotations"]:
        recs[an["image_id"].split("-", 1)[1].lower()][an["image_id"]] += 1
    cases = sorted(c for c in stem_of if stem_of[c] in recs)
    assert len(cases) == 707, len(cases)
    k = np.array([sum(1 for n in recs[stem_of[c]].values() if n > 0) for c in cases], dtype=np.int64)
    G = np.array([sum(recs[stem_of[c]].values()) for c in cases], dtype=np.int64)
    return {"cases": np.array(cases), "coco_k": k, "coco_G": G}


def write(path: str | os.PathLike = None) -> str:
    path = str(path or fp.WORK / TABLE)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **build())
    return path


def load_from(path) -> dict:
    """{cases, coco_k, coco_G} from a table written by `write()`."""
    z = np.load(path)
    return {k: z[k] for k in ("cases", "coco_k", "coco_G")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    print("wrote", write(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
