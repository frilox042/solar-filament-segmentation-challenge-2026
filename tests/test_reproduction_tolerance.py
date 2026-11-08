"""Item: the selector step judges a CSV by one reproduction rule; bit-identity is reported, never required.

  PASS iff at most 3 images whose object count differs, each by at most 1 object, AND the total object count within
  +-3, AND the pooled union-mask Dice over the images whose count matches >= the floor (and verify_submission passes).
  Reported as information: md5_matches_reference, same_count_everywhere, the all-image pooled Dice, worst images.

Each clause is tested on plain numbers (`classify`), so that a control fails for the reason it names and not for
another clause. Negative controls (must FAIL): 4 images off by one; one image off by 2; pooled Dice just below the
floor; a total off by 4; an unknown image id; verify_submission failing. Positive controls: 3 images +-1 at the floor.
The real nnHALO fresh-clone re-run (tests/fixtures/reruns/) is classified as the README records.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
pytest.importorskip("pycocotools")

from filament_seg import config as C  # noqa: E402
from filament_seg import selector_stage as S  # noqa: E402
from filament_seg.submission import compare  # noqa: E402
from filament_seg.submission.rle import rle_for  # noqa: E402

RULE = C.load("FAST").tolerance_rule
FLOOR = RULE["min_pooled_dice_count_matched"]
REF_ROWS = 1185                                    # the size of submissions/FAST.csv


def _cls(count_diff, dice, rows=REF_ROWS, ref_rows=REF_ROWS):
    return compare.classify(RULE, count_diff, rows, ref_rows, dice)


# ---- each clause on plain numbers -------------------------------------------------------------------------------

def test_identical_counts_and_dice_pass_and_report_same_count():
    r = _cls({}, 1.0)
    assert r["within_tolerance"] and not r["failures"] and r["same_count_everywhere"]


def test_three_images_off_by_one_at_the_floor_passes():
    r = _cls({"a": 1, "b": -1, "c": 1}, FLOOR, rows=REF_ROWS + 1)
    assert r["within_tolerance"] and not r["failures"] and not r["same_count_everywhere"]


def test_four_images_off_by_one_fails():
    r = _cls({"a": 1, "b": -1, "c": -1, "d": 1}, 1.0)
    assert not r["within_tolerance"] and len(r["failures"]) == 1 and "on 4 images" in r["failures"][0]


def test_one_image_off_by_two_fails():
    r = _cls({"a": 2}, 1.0, rows=REF_ROWS + 2)
    assert not r["within_tolerance"] and len(r["failures"]) == 1 and "differs by 2" in r["failures"][0]


def test_pooled_dice_below_the_floor_fails():
    r = _cls({}, FLOOR - 1e-4)
    assert not r["within_tolerance"] and len(r["failures"]) == 1 and "count-matched" in r["failures"][0]
    assert r["same_count_everywhere"]                                   # information only: does not rescue it


def test_total_off_by_four_fails():
    """The total clause on its own (3 images +1 each, and a reference that is itself 1 object short)."""
    r = _cls({"a": 1, "b": 1, "c": 1}, 1.0, rows=104, ref_rows=100)
    assert not r["within_tolerance"] and len(r["failures"]) == 1 and "total object count" in r["failures"][0]
    assert _cls({"a": 1, "b": 1, "c": 1}, 1.0, rows=103, ref_rows=100)["within_tolerance"]   # +-3 exactly passes


def test_structural_failure_fails():
    r = compare.classify(RULE, {}, REF_ROWS, REF_ROWS, 1.0, ["image ids not in the test set"])
    assert not r["within_tolerance"] and not r["same_count_everywhere"]


# ---- end to end on synthetic CSVs ---------------------------------------------------------------------------------

IDS = ["imgA", "imgB", "imgC"]          # imgC: empty in both files


def _box(y, x, h, w):
    m = np.zeros((2048, 2048), bool)
    m[y:y + h, x:x + w] = True
    return m


def _csv(path, objs):
    lines = ["filament_id,segmentation_rle"]
    for img, masks in objs.items():
        lines += [f"{img}_{n},{rle_for(m)}" for n, m in enumerate(masks, start=1)]
    path.write_text("\n".join(lines) + "\n")
    return path


REF = {"imgA": [_box(100, 100, 40, 200), _box(500, 500, 30, 300)], "imgB": [_box(1000, 1000, 50, 50)]}


@pytest.fixture
def ref(tmp_path):
    return _csv(tmp_path / "ref.csv", REF)


def test_identical_passes_and_md5_matches(tmp_path, ref):
    r = compare.compare(_csv(tmp_path / "a.csv", REF), ref, IDS, RULE)
    assert r["within_tolerance"] and r["md5_matches_reference"] and r["same_count_everywhere"]
    assert r["min_image_dice"] == 1.0 and r["pooled_dice_count_matched"] == r["pooled_dice_all"] == 1.0


def test_one_column_wider_passes_and_md5_differs(tmp_path, ref):
    grown = _box(100, 100, 40, 200)
    grown[100, 300] = True                                                # one pixel more: same counts, Dice ~0.99997
    near = dict(REF, imgA=[grown, REF["imgA"][1]])
    r = compare.compare(_csv(tmp_path / "a.csv", near), ref, IDS, RULE)
    assert r["within_tolerance"] and not r["md5_matches_reference"] and r["same_count_everywhere"]
    assert FLOOR < r["pooled_dice_count_matched"] < 1.0 and r["worst_image"] == "imgA"
    out = {"apply": {"verify_rc": 0, "within_tolerance": r["within_tolerance"],
                     "md5_matches_reference": r["md5_matches_reference"]}}
    assert S.status(out) == 0                                             # md5 mismatch is NOT a failure


def test_count_change_is_excluded_from_the_matched_dice(tmp_path, ref):
    """A dropped object on imgA changes that image's count: the rule passes on the matched images (Dice 1) while the
    all-image pooled Dice, reported as information, is far below the floor."""
    dropped = dict(REF, imgA=[REF["imgA"][0]])
    r = compare.compare(_csv(tmp_path / "a.csv", dropped), ref, IDS, RULE)
    assert r["images_count_differs"] == {"imgA": -1} and not r["same_count_everywhere"]
    assert r["within_tolerance"] and r["pooled_dice_count_matched"] == 1.0 and r["pooled_dice_all"] < FLOOR
    assert r["worst_image"] == "imgA" and r["worst_matched_image"] in ("imgB", "imgC")


def test_shifted_object_fails_on_pooled_dice(tmp_path, ref):
    shifted = dict(REF, imgB=[_box(1000, 1020, 50, 50)])                  # same count, much lower Dice
    r = compare.compare(_csv(tmp_path / "a.csv", shifted), ref, IDS, RULE)
    assert not r["within_tolerance"] and not r["images_count_differs"]
    assert any("count-matched" in f for f in r["failures"])
    assert S.status({"apply": {"verify_rc": 0, "within_tolerance": r["within_tolerance"]}}) == 1


def test_unknown_image_id_fails(tmp_path, ref):
    r = compare.compare(_csv(tmp_path / "a.csv", dict(REF, imgZ=[_box(0, 0, 30, 30)])), ref, IDS, RULE)
    assert not r["within_tolerance"] and r["unknown_image_ids"] == ["imgZ"]


def test_verify_failure_fails_even_within_tolerance():
    assert S.status({"apply": {"verify_rc": 1, "within_tolerance": True}}) == 1
    fit_bad = {"fit": {"two_cut_matches_config": False}, "apply": {"verify_rc": 0, "within_tolerance": True}}
    assert S.status(fit_bad) == 1


# ---- the config and the real files ------------------------------------------------------------------------------

def test_slot_config_states_the_rule():
    want = {"max_images_count_differs": 3, "max_count_diff_per_image": 1, "max_total_count_diff": 3,
            "min_pooled_dice_count_matched": FLOOR}
    j = C.load_json("FAST")["reproduction_tolerance"]
    assert j["rule"] == want and "tiers" not in j
    cfg = C.load("FAST")
    assert cfg.tolerance_rule == want and cfg.reference_csv == ROOT / "submissions" / "FAST.csv"
    assert len(C.test_image_ids()) == 180


def test_uploaded_file_against_itself():
    """The real reference compared with itself: passes, bit-identical, rows = configs expected_rows."""
    ref = ROOT / "submissions" / "FAST.csv"
    r = compare.compare(ref, ref, C.test_image_ids(), RULE)
    assert r["within_tolerance"] and r["md5_matches_reference"] and r["md5"] == C.load_json("FAST")["expected_md5"]
    assert r["rows"] == C.load_json("FAST")["expected_rows"] and not r["unknown_image_ids"]
    json.dumps(r)


RERUNS = ROOT / "tests" / "fixtures" / "reruns"


def test_real_fresh_clone_rerun_passes_as_recorded():
    """FAST, fresh clone, same laptop GPU (2026-11-02): two borderline selector decisions flipped (README
    "Reproduction tolerance", "Measured runs"); not bit-identical, counts differ on 2 images, the rule passes."""
    assert sorted(p.name for p in RERUNS.iterdir()) == ["FAST_fresh_clone.csv"]
    r = compare.compare(RERUNS / "FAST_fresh_clone.csv", ROOT / "submissions" / "FAST.csv", C.test_image_ids(), RULE)
    assert r["images_count_differs"] == {"20220523234912Lh": 1, "20220621185332Bh": -1} and r["rows"] == 1185
    assert not r["md5_matches_reference"] and not r["same_count_everywhere"]
    assert r["within_tolerance"], r["failures"]
    assert r["pooled_dice_count_matched"] >= 0.9999 and r["pooled_dice_all"] < 0.999      # the all-image value fails
