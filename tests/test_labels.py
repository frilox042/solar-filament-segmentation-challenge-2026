"""Training-label builder (`filament_seg.prep.labels`, `scripts/build_labels.py`): synthetic unit tests, and the
identity with the shipped members' labelsTr.

The data test needs the competition download (FILAMENT_KAGGLE_ROOT train_images, FILAMENT_KAGGLE_COCO) and a reference
nnUNet_raw holding Dataset005 and Dataset036 labelsTr ($nnUNet_raw); it is skipped when they are not staged (set
FILAMENT_REQUIRE_DATA=1 to make that a failure). The full 707-case gate with both negative controls is
`tests/eval_records/build_labels_check.py`, record `tests/eval_records/build_labels_record.json`.
"""
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament_seg.evaluation import records_coco as RC  # noqa: E402
from filament_seg.prep import labels as L  # noqa: E402


# ------------------------------------------------------------------------------------------------------- barb_label
def _t_shape():
    m = np.zeros((200, 200), bool)
    m[95:105, 20:180] = True            # spine, 160 x 10
    m[40:95, 98:104] = True             # one barb off the middle
    return m


def test_barb_label_t_shape_has_barb_class():
    lab = L.barb_label(_t_shape())
    assert set(np.unique(lab).tolist()) == {0, 1, 2}
    assert ((lab > 0) == _t_shape()).all()                     # classes partition the object, nothing outside it
    assert (lab[40:80, 98:104] == 2).mean() > 0.9              # the barb is class 2
    assert (lab[95:105, 20:60] == 1).all()                     # the far spine is class 1


def test_barb_label_small_object_and_straight_bar():
    small = np.zeros((64, 64), bool)
    small[10:19, 10:30] = True                                  # 180 px < MIN_OBJ
    assert L.barb_label(small)[small].tolist() == [1] * int(small.sum())
    bar = np.zeros((64, 256), bool)
    bar[20:30, 10:240] = True
    lab = L.barb_label(bar)
    assert 2 not in lab and ((lab == 1) == bar).all()
    assert not L.barb_label(np.zeros((32, 32), bool)).any()


def test_spine_and_barbs_line_and_tree():
    line = np.zeros((10, 50), bool)
    line[5, 5:45] = True
    sp, bb = L.spine_and_barbs(line)
    assert (sp == line).all() and not bb.any()
    tree = line.copy()
    tree[1:5, 25] = True
    sp, bb = L.spine_and_barbs(tree)
    assert bb.sum() == 4 and not (sp & bb).any() and ((sp | bb) == tree).all()


# ------------------------------------------------------------------------------------------- end to end, synthetic
def _sq(x0, y0, s):
    return [[x0, y0, x0 + s, y0, x0 + s, y0 + s, x0, y0 + s]]


def _download(tmp_path):
    root = tmp_path / "kaggle"
    imgs = root / L.TRAIN_IMAGES
    imgs.mkdir(parents=True)
    for f in ("b_img.jpeg", "a_img.jpeg"):
        Image.new("L", (2048, 2048)).save(imgs / f)
    # a T: the spine and the barb are two records' rectangles, so only their union is a tree
    coco = {"images": [{"id": "010101-a_img", "file_name": "a_img.jpeg"},
                       {"id": "020102-a_img", "file_name": "a_img.jpeg"},
                       {"id": "010101-b_img", "file_name": "b_img.jpeg"}],
            "annotations": [{"id": 1, "image_id": "010101-a_img",
                             "segmentation": [[20, 95, 180, 95, 180, 105, 20, 105]]},
                            {"id": 2, "image_id": "020102-a_img",
                             "segmentation": [[98, 40, 104, 40, 104, 95, 98, 95]]},
                            {"id": 3, "image_id": "010101-b_img", "segmentation": _sq(500, 500, 8)}]}
    (tmp_path / "coco.json").write_text(json.dumps(coco))
    cfg = tmp_path / "configs"
    cfg.mkdir()
    (cfg / "case_mapping_full.json").write_text(json.dumps({"train": {"img_0000": "a_img.jpeg",
                                                                      "img_0001": "b_img.jpeg"}}))
    return root, tmp_path / "coco.json", cfg


def test_build_end_to_end_synthetic(tmp_path, monkeypatch):
    root, coco, cfg = _download(tmp_path)
    monkeypatch.setattr(L.fp, "CONFIGS", cfg)
    RC._records_by_stem.cache_clear()
    raw = tmp_path / "raw"
    r = L.build(list(L.DATASETS), coco, root, raw, workers=1)
    assert r["cases"] == 2 and r["files_written"] == 8
    u = np.asarray(Image.open(raw / L.DATASETS["D005"] / "labelsTr" / "img_0000.png"))
    assert u.dtype == np.uint8 and set(np.unique(u).tolist()) == {0, 1}
    want = RC.union_mask(RC.records_by_case(cfg / "case_mapping_full.json", coco)["img_0000"])
    assert np.array_equal(u > 0, want)
    for d in ("D007", "D012"):
        assert np.array_equal(np.asarray(Image.open(raw / L.DATASETS[d] / "labelsTr" / "img_0000.png")), u)
    b = np.asarray(Image.open(raw / L.DATASETS["D036"] / "labelsTr" / "img_0000.png"))
    assert np.array_equal(b, L.barb_label(want)) and 2 in b and np.array_equal(b > 0, want)
    # resume-safe: nothing rewritten; verify against what was written passes, and fails on a corrupted file
    assert L.build(list(L.DATASETS), coco, root, raw, workers=1)["files_written"] == 0
    assert all(v["D036"]["equal"] for v in L.verify(["D036"], raw, coco, root, workers=1).values())
    p = raw / L.DATASETS["D036"] / "labelsTr" / "img_0001.png"
    bad = np.asarray(Image.open(p)).copy()
    bad[0, 0] = 1
    Image.fromarray(bad, mode="L").save(p)
    res = L.verify(["D036"], raw, coco, root, workers=1)
    assert res["img_0001"]["D036"] == {"equal": False, "n_diff": 1, "fg": int((bad > 0).sum()), "barb": 0}
    assert res["img_0000"]["D036"]["equal"]


def test_enumeration_must_match_mapping(tmp_path, monkeypatch):
    root, coco, cfg = _download(tmp_path)
    monkeypatch.setattr(L.fp, "CONFIGS", cfg)
    Image.new("L", (2048, 2048)).save(root / L.TRAIN_IMAGES / "0_extra.jpeg")   # sorts first: renumbers every case
    with pytest.raises(ValueError, match="does not reproduce"):
        L.enumerate_cases(root)
    (root / L.TRAIN_IMAGES / "0_extra.jpeg").unlink()
    Image.new("L", (1024, 1024)).save(root / L.TRAIN_IMAGES / "a_img.jpeg")
    with pytest.raises(ValueError, match="image size"):
        L.enumerate_cases(root)


# ------------------------------------------------------------------------------------------------------- real data
def _data_ready():
    from filament_seg import filament_paths as fp
    importlib.reload(fp)            # another test may have re-pointed the settings module at a temporary WORK
    importlib.reload(L)
    need = [fp.KAGGLE_COCO, fp.KAGGLE_ROOT / L.TRAIN_IMAGES, fp.DATASET_RAW[5] / "labelsTr",
            fp.DATASET_RAW[36] / "labelsTr"]
    missing = [str(x) for x in need if not Path(x).exists()]
    if missing and os.environ.get("FILAMENT_REQUIRE_DATA") == "1":
        pytest.fail(f"FILAMENT_REQUIRE_DATA=1 but missing {missing}")
    return not missing


def test_built_labels_equal_shipped_labelsTr():
    if not _data_ready():
        pytest.skip("competition download / Dataset005+036 labelsTr not staged")
    RC._records_by_stem.cache_clear()
    ref = L.fp.NNUNET_RAW
    union = L.verify(["D005"], ref)
    assert len(union) == 707 and all(v["D005"]["equal"] for v in union.values())
    barb = L.verify(["D036"], ref, limit=64)                     # the full 707 run is the gate record
    assert len(barb) == 64 and all(v["D036"]["equal"] for v in barb.values())
    assert sum(v["D036"]["barb"] for v in barb.values()) > 0
    # negative control: one record dropped breaks exactly its observation, in both label kinds
    cases, recs = L.records()
    case = next(c for c in sorted(recs) if len(recs[c]) >= 2)
    nc = L.verify(["D005", "D036"], ref, limit=8, recs_override=dict(recs, **{case: recs[case][1:]}))
    assert sorted(c for c, v in nc.items() if not v["D005"]["equal"]) == [case]
    assert sorted(c for c, v in nc.items() if not v["D036"]["equal"]) == [case]
