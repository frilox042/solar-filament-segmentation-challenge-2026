"""Item: the default predict path APPLIES the released selector bag of nnHALO; refitting is an explicit option.

Checks: the released bag is listed in weights/MANIFEST.sha256 and matches it; it loads to 5 regressors with the
config's cut; `apply` defaults to it and `both` applies its OWN fitted bag; the plans of `--from-images` and of the
CPU-only entry point default to `apply`, wire `--bag-dir`, and put `selector_tables` before the fit under `--refit`.
Only one bag ships (FAST), and the CLIs accept no other configuration.
Negative control: a bag whose bytes differ from the manifest is refused BEFORE it is unpickled.
"""
import hashlib
import importlib
import importlib.util
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


def _reload(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    from filament_seg import filament_paths
    importlib.reload(filament_paths)
    from filament_seg import config, selector_stage
    importlib.reload(config)
    return importlib.reload(selector_stage)


def test_released_bag_matches_manifest_and_config(monkeypatch):
    pytest.importorskip("sklearn")
    slot = "FAST"
    S = _reload(monkeypatch)
    p = ROOT / "weights" / "selector" / f"{slot}_bag.pkl"
    assert S.released_bag_path(slot) == p
    assert hashlib.sha256(p.read_bytes()).hexdigest() == S.manifest_sha256(f"selector/{slot}_bag.pkl")
    d = S.load_released_bag(slot)
    two_cut = json.loads((ROOT / "configs" / f"{slot}.json").read_text())["selector"]["expected_two_cut"]
    assert len(d["bag"]) == 5 and 2 * d["cut"] == two_cut and d["info"]["two_cut"] == two_cut
    assert d["info"]["slot"] == slot and "fit_s" not in d["info"]      # deterministic file: no timing inside


def test_only_the_fast_bag_ships(monkeypatch):
    assert sorted(p.name for p in (ROOT / "weights" / "selector").iterdir()) == ["FAST_bag.pkl"]
    S = _reload(monkeypatch)
    assert sorted(p.name for p in (ROOT / "configs").glob("*.json")) == ["FAST.json", "case_mapping_full.json"]
    for cmd in ("fit", "apply", "both"):
        with pytest.raises(SystemExit):
            S.main([cmd, "--slot", "OTHER"])


def test_tampered_bag_is_refused_before_unpickling(tmp_path, monkeypatch):
    marker = tmp_path / "EXECUTED"

    class Boom:
        def __reduce__(self):
            return (open, (str(marker), "w"))                            # would create the marker if unpickled

    (tmp_path / "FAST_bag.pkl").write_bytes(pickle.dumps(Boom()))
    S = _reload(monkeypatch, FILAMENT_SELECTOR_BAGS=tmp_path)
    with pytest.raises(ValueError, match="refusing to load"):
        S.load_released_bag("FAST")
    assert not marker.exists()


def test_missing_manifest_entry_is_an_error(tmp_path, monkeypatch):
    S = _reload(monkeypatch)
    m = tmp_path / "M.sha256"
    m.write_text("# nothing\n" + "0" * 64 + "  selector/OTHER_bag.pkl\n")
    with pytest.raises(KeyError):
        S.manifest_sha256("selector/FAST_bag.pkl", m)


def test_apply_defaults_to_released_and_both_applies_its_own_fit(tmp_path, monkeypatch):
    S = _reload(monkeypatch, FILAMENT_WORK=tmp_path / "work")
    seen = []
    monkeypatch.setattr(S, "run_fit", lambda cfg: {"two_cut_matches_config": True})
    monkeypatch.setattr(S, "run_apply", lambda cfg, csv, bag=None: seen.append(bag) or
                        {"verify_rc": 0, "within_tolerance": True})
    assert S.main(["apply", "--slot", "FAST", "--csv", str(tmp_path / "a.csv")]) == 0
    assert S.main(["apply", "--slot", "FAST", "--csv", str(tmp_path / "a.csv"), "--bag", "X.pkl"]) == 0
    assert S.main(["both", "--slot", "FAST", "--csv", str(tmp_path / "a.csv")]) == 0
    from filament_seg import config as C
    assert seen == [None, "X.pkl", S.bag_path(C.load("FAST"))]
    with pytest.raises(SystemExit):
        S.main(["both", "--slot", "FAST", "--bag", "X.pkl"])


def _plan(tmp_path, *extra):
    env = dict(os.environ, FILAMENT_WORK=str(tmp_path / "work"), FILAMENT_WEIGHTS=str(tmp_path / "weights"))
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "predict.py"), "--from-images", "--dry-run", *extra],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    return [json.loads(x) for x in r.stdout.splitlines() if x.startswith("{")]


def test_from_images_default_applies_released(tmp_path):
    sel = [d for d in _plan(tmp_path) if d["step"].startswith("selector")]      # --slot defaults to FAST
    assert [d["step"] for d in sel] == ["selector_FAST"]
    for d in sel:
        assert d["argv"][:3] == ["-m", "filament_seg.selector_stage", "apply"] and "--bag" not in d["argv"]


def test_from_images_bag_dir_is_wired(tmp_path):
    d = next(d for d in _plan(tmp_path, "--slot", "FAST", "--bag-dir", str(tmp_path / "bags"))
             if d["step"] == "selector_FAST")
    assert d["argv"][2] == "apply" and d["argv"][d["argv"].index("--bag") + 1] == str(tmp_path / "bags" / "FAST_bag.pkl")


def test_from_images_refit_builds_tables_then_fits(tmp_path):
    steps = _plan(tmp_path, "--slot", "FAST", "--refit")
    names = [d["step"] for d in steps]
    assert names[-2:] == ["selector_tables", "selector_FAST"]
    assert steps[-2]["argv"] == ["-m", "filament_seg.selector_tables"]
    assert steps[-1]["argv"][2] == "both" and "--bag" not in steps[-1]["argv"]


@pytest.mark.parametrize("args", [["--slot", "OTHER"], ["--slot", "all"], ["--from-maps", "X"]])
def test_predict_accepts_only_fast(tmp_path, args):
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "predict.py"), "--from-images", "--dry-run", *args],
                       capture_output=True, text=True)
    assert r.returncode == 2, r.stdout


def test_refit_and_bag_dir_are_exclusive(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "predict.py"), "--slot", "FAST", "--refit",
                        "--bag-dir", str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 2 and "not allowed" in r.stderr


def test_cpu_only_entry_point_defaults_to_apply(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("predict_cli", ROOT / "scripts" / "predict.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    calls = []

    class R:
        returncode = 0

    monkeypatch.setattr(m.subprocess, "run", lambda argv, **kw: calls.append(argv[3:]) or R())
    assert m.main(["--slot", "FAST"]) == 0
    assert [c[0] for c in calls] == ["apply"] and all("--bag" not in c for c in calls)
    assert calls[0][1:3] == ["--slot", "FAST"]
    calls.clear()
    assert m.main(["--slot", "FAST", "--refit"]) == 0 and calls[0][0] == "both"
