"""The training recipe: its splits rule reproduces the shipped splits (SHA-256 of folds 0-4 recorded from
nnUNet_preprocessed/<Dataset>/splits_final.json of the training run), every script/module it calls starts, its output
paths are the ones the selector reads (configs/FAST.json), and it never writes into the shipped weights."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from filament_seg import filament_paths as fp  # noqa: E402
from filament_seg import training as T  # noqa: E402


def test_splits_rule_reproduces_shipped_splits():
    assert T.splits_sha256(T.make_splits(T.train_cases())) == T.SPLITS_SHA256


def test_splits_negative_control():
    """A different fold count (or unsorted order) must NOT match the recorded hash."""
    assert T.splits_sha256(T.make_splits(T.train_cases(), k=4)) != T.SPLITS_SHA256
    rev = list(reversed(T.train_cases()))
    folds = [rev[i::5] for i in range(5)]
    bad = [{"train": sorted(c for c in rev if c not in set(f)), "val": sorted(f)} for f in folds]
    assert T.splits_sha256(bad) != T.SPLITS_SHA256


def test_full_recipe_covers_both_networks_and_five_folds():
    names = [s.name for s in T.full_recipe(Path("/r"))]
    for ds in T.DATASETS.values():
        assert sum(n.startswith(f"train {ds} fold") for n in names) == 5
    assert names[-1] == "selector fit"
    assert "selector fit" not in [s.name for s in T.smoke_full_recipe(Path("/r"))]       # needs all 707 observations


def _entry(step):
    """(path-or-module, is_module) of a step's entry point, or None for an nnU-Net -c call."""
    a = step.argv
    if a[0] == "-c":
        return None
    if a[0] == "-m":
        return a[1], True
    return a[0], False


STEPS = [s for s in T.full_recipe(Path("/r")) + T.smoke_full_recipe(Path("/r")) + T.smoke_recipe(Path("/r"))
         if _entry(s)]


@pytest.mark.parametrize("step", STEPS, ids=lambda s: s.name)
def test_every_recipe_entry_point_starts(step, tmp_path):
    """`--help` of each script / module the recipe calls exits 0 in the step's environment. Catches imports that
    only work from another directory (the out-of-fold tools import `filament_paths` as a top-level module)."""
    target, is_mod = _entry(step)
    if "predict_oof_probs" in target and importlib.util.find_spec("torch") is None:
        pytest.skip("predict_oof_probs needs torch (the nnU-Net environment)")
    argv = [sys.executable, *(["-m", target] if is_mod else [target]), "--help"]
    r = subprocess.run(argv, cwd=str(ROOT), env=T.env(tmp_path), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-1500:]


def test_entry_point_check_negative_control(tmp_path):
    """The same check must FAIL for the broken form the recipe used to have (`-m` on a frozen out-of-fold tool)."""
    r = subprocess.run([sys.executable, "-m", "filament_seg.gpu.make_oof_set", "--help"], cwd=str(ROOT),
                       env=T.env(tmp_path), capture_output=True, text=True)
    assert r.returncode != 0 and "filament_paths" in r.stderr


def test_recipe_outputs_are_what_the_selector_reads():
    cfg = json.loads((fp.CONFIGS / "FAST.json").read_text())["cpu_stage"]
    root = Path("/r")
    argv = {s.name: s.argv for s in T.full_recipe(root)}
    oof_maps = argv["out-of-fold probability maps (D007)"]
    assert oof_maps[oof_maps.index("--out") + 1] == str(root / cfg["oof_prob_dir"])
    d036 = [m for m in cfg["members"] if m["name"] == "D036"][0]
    masks = argv["out-of-fold masks (D036)"]
    assert masks[masks.index("--out") + 1] == str(root / d036["oof_dir"])


def test_training_never_targets_shipped_weights(tmp_path):
    shipped = Path(fp.WEIGHTS).resolve()
    res = T.results_dir(tmp_path).resolve()
    assert shipped not in res.parents and res != shipped
    bad = T.results_dir(shipped / "x").resolve()      # negative control: a root inside the weights tree is caught
    assert shipped in bad.parents
