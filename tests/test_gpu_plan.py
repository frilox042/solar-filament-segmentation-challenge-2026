"""CPU-only dry-run tests of the `--from-images` chain (no GPU, nothing predicted).

Always run: argument parsing and the plan (steps, interpreters, every path under FILAMENT_WORK / FILAMENT_WEIGHTS,
--limit drops the CPU stage and sets FILAMENT_LIMIT, resume = a complete step is skipped).
Run when data is available (otherwise skipped):
  FILAMENT_TEST_KAGGLE_ROOT + FILAMENT_TEST_NNUNET_RAW   one train + one test image's channels rebuilt from the JPEG
                                                          must be array-equal to the stored channels
  FILAMENT_TEST_WEIGHTS_NNUNET + FILAMENT_TEST_PY_NNUNET  nnUNetPredictor built on CPU from a shipped member (1 fold)
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _plan_lines(tmp_path, *extra):
    env = dict(os.environ, FILAMENT_WORK=str(tmp_path / "work"), FILAMENT_WEIGHTS=str(tmp_path / "weights"),
               FILAMENT_SUBMISSIONS_OUT=str(tmp_path / "work" / "out"))    # not the repo's submissions/out (a notebook run fills it)
    for k in ("FILAMENT_WEIGHTS_NNUNET", "nnUNet_raw", "nnUNet_results"):
        env.pop(k, None)
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "predict.py"), "--from-images", "--dry-run", *extra],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    return [json.loads(x) for x in r.stdout.splitlines() if x.startswith("{")]


def test_plan_fast(tmp_path):
    lines = _plan_lines(tmp_path, "--slot", "FAST")
    steps = [(d["slot"], d["step"]) for d in lines]
    assert steps == [("FAST", "channels"), ("FAST", "base_D007_f0"), ("FAST", "bin_D036_f0"),
                     ("FAST", "selector_FAST")]               # no GPU step for the pseudo-member P (CPU stage)
    assert lines == _plan_lines(tmp_path)                 # --slot defaults to FAST
    work, weights = str(tmp_path / "work"), str(tmp_path / "weights")
    for d in lines:
        assert not d["complete"]
        assert d["out"].startswith(work) or d["out"].startswith(str(ROOT / "submissions")), d
        for a in d["argv"]:
            if a.startswith("/") and "/gpu" not in a and "submissions" not in a:
                assert a.startswith(work) or a.startswith(weights), (d["step"], a)
    fast_d007 = next(d for d in lines if d["step"] == "base_D007_f0")
    assert "--save-probabilities" in fast_d007["argv"] and fast_d007["argv"][fast_d007["argv"].index("--folds") + 1] == "0"
    assert {d["env"] for d in lines} == {"cpu", "nnunet"}       # two interpreters
    assert not (tmp_path / "work").exists()          # a dry run writes nothing


def test_only_fast_is_planned(tmp_path, monkeypatch):
    pipeline = _pipeline(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        pipeline.plan("OTHER")
    assert not hasattr(pipeline, "_map_steps")


def test_limit_skips_cpu_stage(tmp_path):
    lines = _plan_lines(tmp_path, "--slot", "FAST", "--limit", "2")
    assert [d["step"] for d in lines] == ["channels", "base_D007_f0", "bin_D036_f0"]
    assert "--limit" in lines[0]["argv"] and any("limit/2" in a for a in lines[1]["argv"])


def test_resume_skips_complete_step(tmp_path, monkeypatch):
    monkeypatch.setenv("FILAMENT_WORK", str(tmp_path / "work"))
    sys.path.insert(0, str(SRC))
    import importlib
    from filament_seg import filament_paths
    importlib.reload(filament_paths)
    from filament_seg.gpu import pipeline
    importlib.reload(pipeline)
    st = pipeline.plan("FAST", limit=2)[2]           # bin_D036_f0
    st.out.mkdir(parents=True)
    assert not st.done()
    for c in ("filamentTs_0000", "filamentTs_0001"):
        (st.out / f"{c}.png").write_bytes(b"x")
    assert st.done()


def _pipeline(tmp_path, monkeypatch):
    monkeypatch.setenv("FILAMENT_WORK", str(tmp_path / "work"))
    for k in ("nnUNet_raw", "FILAMENT_PREDICTIONS"):
        monkeypatch.delenv(k, raising=False)
    sys.path.insert(0, str(SRC))
    import importlib
    from filament_seg import filament_paths
    importlib.reload(filament_paths)
    from filament_seg.gpu import pipeline
    return importlib.reload(pipeline)


def _touch(paths):
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")


def test_channels_step_checks_every_dataset_of_the_slot(tmp_path, monkeypatch):
    """FAST's channel step requires exactly the two networks' inputs (D007, D036) and no longer D005, which FAST never
    reads. With D007 built but D036 missing it must NOT be complete, and must list exactly the D036 files as missing."""
    pipeline = _pipeline(tmp_path, monkeypatch)
    st = pipeline.plan("FAST", limit=2)[0]
    assert st.name == "channels"
    # channel counts per dataset: D007 5, D036 6 -> 2 * 11; D005 (4 channels) is not part of FAST's step
    assert len(st.required) == 2 * 11
    assert not any("Dataset005_FilamentFull" in str(p) for p in st.required)
    _touch([p for p in st.required if "Dataset036_LimbDarkBarbClass" not in str(p)])
    assert not st.done()
    missing = st.missing()
    assert len(missing) == 2 * 6 and all("Dataset036_LimbDarkBarbClass" in str(p) for p in missing), missing
    _touch(missing)
    assert st.done()


def test_channels_step_one_missing_channel_is_incomplete(tmp_path, monkeypatch):
    """Negative control: every file but ONE channel of one case (not the old *_0005 probe) leaves the step undone."""
    pipeline = _pipeline(tmp_path, monkeypatch)
    st = pipeline.plan("FAST", limit=2)[0]
    _touch(st.required)
    victim = next(p for p in st.required if p.name.endswith("_0002.png") and "Dataset007" in str(p))
    victim.unlink()
    assert not st.done() and st.missing() == [victim]


def test_run_reruns_incomplete_channels(tmp_path, monkeypatch):
    """`run` itself executes the channel step when one dataset is missing (subprocess stubbed, channel step only),
    and skips it once every dataset is complete."""
    pipeline = _pipeline(tmp_path, monkeypatch)
    orig_plan, calls = pipeline.plan, []
    _touch([p for p in orig_plan("FAST", 2)[0].required if "Dataset036" not in str(p)])

    class R:
        returncode = 0

    monkeypatch.setattr(pipeline.subprocess, "run", lambda argv, **kw: calls.append(argv) or R())
    monkeypatch.setattr(pipeline, "plan", lambda slot, limit=0, device="cuda", **kw: orig_plan(slot, limit, device)[:1])
    assert pipeline.run(["FAST"], limit=2) == 0
    assert len(calls) == 1 and "D036" in calls[0], calls
    _touch(orig_plan("FAST", 2)[0].missing())
    calls.clear()
    assert pipeline.run(["FAST"], limit=2) == 0 and calls == []


@pytest.mark.skipif(not (os.environ.get("FILAMENT_TEST_KAGGLE_ROOT") and os.environ.get("FILAMENT_TEST_NNUNET_RAW")),
                    reason="needs the competition download and a stored nnUNet_raw tree")
def test_channels_equal_stored_one_image_per_split():
    code = ("import sys, json; sys.path.insert(0, %r); from filament_seg.prep import prepare; "
            "print(json.dumps(prepare.check(1, %r, %r)))" % (str(SRC), os.environ["FILAMENT_TEST_KAGGLE_ROOT"],
                                                             os.environ["FILAMENT_TEST_NNUNET_RAW"]))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    res = json.loads(r.stdout.strip().splitlines()[-1])
    assert len(res) == 2 * (4 + 1 + 5 + 6) and all(res.values()), res


@pytest.mark.skipif(not (os.environ.get("FILAMENT_TEST_WEIGHTS_NNUNET") and os.environ.get("FILAMENT_TEST_PY_NNUNET")),
                    reason="needs nnU-Net weights and the nnU-Net interpreter")
def test_nnunet_predictor_constructs_on_cpu():
    md = Path(os.environ["FILAMENT_TEST_WEIGHTS_NNUNET"]) / "Dataset007_FilamentVessel" / \
        "nnUNetTrainer_250epochs__nnUNetPlans__2d"
    r = subprocess.run([os.environ["FILAMENT_TEST_PY_NNUNET"], "-m", "filament_seg.gpu.nnunet_binaries",
                        "--model-dir", str(md), "--folds", "0", "--device", "cpu", "--construct-only"],
                       capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=str(SRC)))
    assert r.returncode == 0 and "OK predictor: 1 fold(s)" in r.stdout, r.stdout + r.stderr


def test_failed_selector_step_is_rerun_not_skipped(tmp_path, monkeypatch):
    """The selector writes its CSV before verify/tolerance run, so a failed step (rc 1) leaves a CSV behind. The step
    must still count as NOT done (success marker missing) and be executed again; after an rc-0 run it is skipped."""
    monkeypatch.setenv("FILAMENT_SUBMISSIONS_OUT", str(tmp_path / "out"))
    pipeline = _pipeline(tmp_path, monkeypatch)
    orig_plan = pipeline.plan
    st = orig_plan("FAST")[-1]
    assert st.name == "selector_FAST"
    calls, rcs = [], [1, 0]

    class R:
        def __init__(self, rc):
            self.returncode = rc

    def fake_run(argv, **kw):
        calls.append(argv)
        (tmp_path / "out" / "FAST.csv").write_text("filament_id,segmentation_rle\n")   # written before the check
        return R(rcs.pop(0))

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    monkeypatch.setattr(pipeline, "plan", lambda slot, limit=0, device="cuda", **kw: orig_plan(slot, limit, device)[-1:])
    assert pipeline.run(["FAST"]) == 1 and len(calls) == 1
    assert (tmp_path / "out" / "FAST.csv").exists() and not st.done()
    assert st.out.is_dir() and len(list(st.out.glob(st.pattern))) >= st.count   # the old CSV-only rule: "done"
    assert pipeline.run(["FAST"]) == 0 and len(calls) == 2                      # re-executed, now passes
    assert st.done()
    assert pipeline.run(["FAST"]) == 0 and len(calls) == 2                      # complete: skipped
