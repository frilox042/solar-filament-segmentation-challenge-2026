"""`selector_tables.build()` needs only the competition COCO and the shipped case mapping: it must run before (and
without) the channels step, i.e. with no nnU-Net tree at all."""
import importlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_builds_without_any_nnunet_tree(tmp_path, monkeypatch):
    cm = json.loads((ROOT / "configs" / "case_mapping_full.json").read_text())["train"]
    stems = sorted(os.path.splitext(v)[0] for v in cm.values())
    images = [{"id": f"r1-{s}"} for s in stems] + [{"id": f"r2-{stems[0]}"}]
    anns = [{"image_id": f"r1-{stems[0]}"}, {"image_id": f"r1-{stems[0]}"}, {"image_id": f"r2-{stems[0]}"},
            {"image_id": f"r1-{stems[5]}"}]
    coco = tmp_path / "coco.json"
    coco.write_text(json.dumps({"images": images, "annotations": anns}))
    monkeypatch.setenv("FILAMENT_KAGGLE_COCO", str(coco))
    monkeypatch.setenv("nnUNet_raw", str(tmp_path / "does_not_exist"))
    monkeypatch.setenv("FILAMENT_WORK", str(tmp_path / "work"))
    sys.path.insert(0, str(ROOT / "src"))
    from filament_seg import filament_paths
    importlib.reload(filament_paths)
    from filament_seg import selector_tables
    T = importlib.reload(selector_tables).build()
    assert len(T["cases"]) == 707 and not (tmp_path / "does_not_exist").exists()
    by_stem = {os.path.splitext(cm[c])[0]: i for i, c in enumerate(T["cases"])}
    i0, i5, i9 = by_stem[stems[0]], by_stem[stems[5]], by_stem[stems[9]]
    assert (T["coco_k"][i0], T["coco_G"][i0]) == (2, 3)
    assert (T["coco_k"][i5], T["coco_G"][i5]) == (1, 1)
    assert (T["coco_k"][i9], T["coco_G"][i9]) == (0, 0)
