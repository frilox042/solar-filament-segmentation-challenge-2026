"""The offline-evaluation gate comparator and the expected-values file (configs/eval/expected.json)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament_seg.evaluation import expected as E  # noqa: E402


def test_gate_comparator_can_fail():
    pub = {"a": 0.3399158122702826, "b": {"c": [1.0, 2.0]}}
    ok = E.compare_tree({"a": 0.3399158122702826, "b": {"c": [1.0, 2.0]}}, pub, 1e-12)
    assert all(v["pass"] for v in ok.values())
    # negative control: a 1e-9 drift and a missing key both fail
    bad = E.compare_tree({"a": 0.3399158122702826 + 1e-9, "b": {}}, pub, 1e-12)
    assert not bad["a"]["pass"] and not bad["b.c"]["pass"]
    assert not E.close([1.0, 2.0], [1.0, 2.0, 3.0], 1.0)["pass"]           # shape mismatch


def test_expected_values_have_source_and_tolerance():
    import json
    j = json.loads((ROOT / "configs" / "eval" / "expected.json").read_text())
    block = j[j["current"]]
    for sec, v in block.items():
        if sec != "model_set":
            assert E.load(sec) == v and v["source"] and 0 < v["tol"] <= 1e-9, sec


def test_records_blocks_png_is_reported_coco_is_regression():
    import json

    import pytest
    j = json.loads((ROOT / "configs" / "eval" / "expected.json").read_text())
    br = j["by_records"]
    assert br["png"] == j["current"]                    # the reported values stay the png block
    coco = j[br["coco"]]
    assert coco["records"] == "coco" and "REGRESSION" in coco["_doc"]
    for sec in ("annotator_agreement", "never_seen"):
        assert E.load(sec, records="coco") == coco[sec] and coco[sec]["source"].startswith("REGRESSION")
        assert E.load(sec, records="png") == j[br["png"]][sec]
        assert coco[sec] != j[br["png"]][sec]           # a separate block, not a copy of the reported one
    with pytest.raises(KeyError):                       # negative control: no block for an unknown source
        E.load("never_seen", records="magfilo")
