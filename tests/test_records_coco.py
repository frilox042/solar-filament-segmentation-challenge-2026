"""The COCO per-record source (`records_coco`): synthetic unit tests, and the 707 / 707 identity with training labels.

The data test needs the competition COCO, the case mapping and Dataset005 labelsTr; it is skipped when they are not
staged (set FILAMENT_REQUIRE_DATA=1 to make that a failure). Its gate record is
`tests/eval_records/records_coco_union_record.json` (`records_coco_check.py union`).
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament_seg.evaluation import host_rows as HR  # noqa: E402
from filament_seg.evaluation import records  # noqa: E402
from filament_seg.evaluation import records_coco as RC  # noqa: E402


def _sq(x0, y0, s):
    return [[x0, y0, x0 + s, y0, x0 + s, y0 + s, x0, y0 + s]]


def _coco(tmp_path, drop_anns_of=None):
    images = [{"id": "020102-img_a", "file_name": "img_a.jpg"}, {"id": "010101-img_a", "file_name": "img_a.jpg"},
              {"id": "010101-img_b", "file_name": "img_b.jpg"}]
    anns = [{"id": 1, "image_id": "020102-img_a", "segmentation": _sq(10, 10, 20), "category_id": 1},
            {"id": 2, "image_id": "010101-img_a", "segmentation": _sq(15, 15, 20), "category_id": 2},
            {"id": 3, "image_id": "010101-img_a", "segmentation": _sq(100, 100, 5), "category_id": 3},
            {"id": 4, "image_id": "010101-img_b", "segmentation": _sq(50, 50, 8), "category_id": 1}]
    anns = [a for a in anns if a["image_id"] != drop_anns_of]
    p = tmp_path / "coco.json"
    p.write_text(json.dumps({"images": images, "annotations": anns}))
    m = tmp_path / "map.json"
    m.write_text(json.dumps({"train": {"img_0001": "img_a.jpg", "img_0002": "IMG_B.jpg"}}))
    return p, m


def test_records_sorted_masks_and_union(tmp_path):
    RC._records_by_stem.cache_clear()
    p, m = _coco(tmp_path)
    by = RC.records_by_case(m, p)
    assert [r.id for r in by["img_0001"]] == ["010101-img_a", "020102-img_a"]       # sorted record id
    assert [r.id for r in by["img_0002"]] == ["010101-img_b"]                         # stems are case-insensitive
    r0 = by["img_0001"][0]
    ref = np.zeros(2048 * 2048, bool)
    for seg in r0.segs:
        ref[HR.decode(seg)] = True
    assert np.array_equal(RC.record_mask(r0), ref.reshape(2048, 2048))
    u = RC.union_mask(by["img_0001"])
    assert np.array_equal(u, RC.record_mask(by["img_0001"][0]) | RC.record_mask(by["img_0001"][1]))
    # negative control: one record less is a different union
    assert not np.array_equal(u, RC.union_mask(by["img_0001"][:1]))


def test_record_without_annotation_is_an_error(tmp_path):
    RC._records_by_stem.cache_clear()
    p, m = _coco(tmp_path, drop_anns_of="010101-img_b")
    with pytest.raises(ValueError, match="no annotation"):
        RC.records_by_case(m, p)


def test_keys_and_source_dispatch():
    assert RC.png_record_key("123-040301-20140609195854bh") == RC.coco_record_key("040301-20140609195854Bh")
    with pytest.raises(ValueError):
        RC.png_record_key("040301-20140609195854Bh")                                    # no export index
    assert records.DEFAULT == "coco" and records.source("coco") is RC
    assert records.source("png").__name__.endswith("records_png")
    with pytest.raises(ValueError):
        records.source("magfilo")


def _data_ready():
    import importlib

    from filament_seg import config as C
    from filament_seg import filament_paths as fp
    importlib.reload(fp)            # another test may have re-pointed the settings module at a temporary WORK
    need = [fp.KAGGLE_COCO, C.train_case_map(), fp.DATASET_RAW[5] / "labelsTr"]
    missing = [str(x) for x in need if not Path(x).exists()]
    if missing and os.environ.get("FILAMENT_REQUIRE_DATA") == "1":
        pytest.fail(f"FILAMENT_REQUIRE_DATA=1 but missing {missing}")
    return not missing


def test_coco_union_equals_training_labels_707():
    if not _data_ready():
        pytest.skip("competition COCO / case mapping / Dataset005 labelsTr not staged")
    from filament_seg import config as C
    spec = importlib.util.spec_from_file_location("rcc", ROOT / "tests" / "eval_records" / "records_coco_check.py")
    rcc = importlib.util.module_from_spec(spec)
    sys.modules["rcc"] = rcc                        # the worker processes unpickle its functions by module name
    spec.loader.exec_module(rcc)
    RC._records_by_stem.cache_clear()
    recs = RC.records_by_case(C.train_case_map())
    assert len(recs) == 707 and sum(len(r) for r in recs.values()) == 1154
    res = rcc.union_identity(recs, rcc.labels_dir())
    assert sum(e for e, _, _ in res.values()) == 707, [c for c, v in res.items() if not v[0]][:5]
    # negative control: drop ONE record from the source; exactly its observation must break
    case, _, nc = rcc.negative_control_drop(recs)
    res_nc = rcc.union_identity(nc, rcc.labels_dir())
    assert sorted(c for c, (e, _, _) in res_nc.items() if not e) == [case]
