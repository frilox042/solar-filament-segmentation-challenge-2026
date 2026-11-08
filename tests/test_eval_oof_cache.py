"""The cross-fitted keeps cache refuses to serve keeps computed from other inputs (stale cache after a retrain)."""
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament_seg.evaluation import oof_selection as OS  # noqa: E402


def _cfg(tmp_path):
    cases = [f"img_{i:04d}" for i in range(7)]
    (tmp_path / "cm.json").write_text(json.dumps({"train": {c: f"{c}.jpg" for c in cases}}))
    for d in ("prob", "m0", "m1"):
        (tmp_path / d).mkdir()
        for c in cases:
            (tmp_path / d / f"{c}.png").write_bytes(f"{d}{c}".encode())
    return SimpleNamespace(slot="FAST", train_case_map=tmp_path / "cm.json", oof_prob_dir=tmp_path / "prob",
                           oof_member_dirs=(tmp_path / "m0", tmp_path / "m1"))


def test_cache_identity_refuses_changed_inputs(tmp_path, monkeypatch):
    monkeypatch.setenv("FILAMENT_EVAL", str(tmp_path / "eval"))
    # isolate from a staged evaluation bundle in the default run root (a notebook run stages one there): with bundled rows
    # the identity is the rows' checksum and the member-file negative control below could not fire
    monkeypatch.setattr(OS.bundle, "rows_path", lambda slot: tmp_path / "no_bundle_rows.npz")
    cfg = _cfg(tmp_path)
    (tmp_path / "eval").mkdir()
    cache = tmp_path / "eval" / "FAST_oof_selection.pkl"
    pickle.dump({"identity": OS.input_identity(cfg), "keeps": "cached"}, open(cache, "wb"))
    assert OS.build(cfg)["keeps"] == "cached"                     # same inputs: served from the cache
    # negative control: one sampled member file changes (as a retrain writing new maps under the same names would)
    (tmp_path / "m1" / "img_0006.png").write_bytes(b"retrained")
    with pytest.raises(RuntimeError, match="other inputs"):
        OS.build(cfg)
