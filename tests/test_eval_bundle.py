"""The evaluation bundle: its level maps preserve every threshold the evaluation applies (and only those), its manifest
refuses a tampered or extra file, and the selector fit refuses a staged level map. No data needed."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament_seg.evaluation import bundle as B  # noqa: E402


def test_levels_preserve_every_threshold_under_both_comparisons():
    v = np.arange(65536, dtype=np.uint16)
    s = B.from_levels(B.to_levels(v))
    p, q = v.astype(np.float32) / np.float32(65535.0), s.astype(np.float32) / np.float32(65535.0)
    for t in (0.1, 0.3, 0.5, 0.92, 0.97):
        assert np.array_equal(p >= t, q >= t) and np.array_equal(p > t, q > t)          # float32 array op
        assert all((float(a) < t) == (float(b) < t) and (float(a) > t) == (float(b) > t)
                   for a, b in zip(p[::7], q[::7], strict=True))                         # Python float scalar op
    # negative control: a threshold the bundle does not keep is NOT preserved (the maps are not probabilities)
    assert not np.array_equal(p >= 0.7, q >= 0.7)


def test_manifest_refuses_tampered_and_extra_files(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "b.bin").write_bytes(b"\x00\x01")
    B.write_manifest(tmp_path)
    assert B.check_manifest(tmp_path) == 2
    (tmp_path / "d" / "b.bin").write_bytes(b"\x00\x02")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        B.check_manifest(tmp_path)
    B.write_manifest(tmp_path)
    (tmp_path / "extra.png").write_bytes(b"")
    with pytest.raises(ValueError, match="unlisted"):
        B.check_manifest(tmp_path)


def test_shipped_bundle_manifest_and_level_table():
    root = ROOT / "eval_bundle"
    if not root.exists():
        pytest.skip("eval_bundle/ not present")
    assert B.check_manifest(root) > 707
    idx = json.loads((root / B.INDEX).read_text())
    assert idx["levels"]["uint16_of_code"] == B.level_table()[1].tolist()


def test_selector_fit_refuses_level_maps(tmp_path):
    from filament_seg import selector_fit
    (tmp_path / B.MARKER).write_text("{}")
    with pytest.raises(RuntimeError, match="threshold-level maps"):
        selector_fit.build_oof_table(SimpleNamespace(oof_prob_dir=tmp_path))
