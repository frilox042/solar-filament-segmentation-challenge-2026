"""End-to-end prediction of Ours-Fast from the raw competition test images: channels -> two nnU-Nets -> CPU selector.

    python scripts/predict.py --from-images --slot FAST [--limit N] [--dry-run] [--device cuda]
                              [--bag-dir DIR | --refit]

Every step is a subprocess with its own interpreter, and every step is RESUME-SAFE: a step whose outputs are already
complete is skipped, and the member predictors skip cases already written. `--limit N` runs the channel and GPU steps
on the first N test cases only (a smoke run); the CPU selector stage needs all 180 and is then skipped.
The CPU stage applies the released selector bag (weights/selector/FAST_bag.pkl) by default; `--refit` re-fits them instead, which
needs the 707-row out-of-fold bundle (the full-retrain tier).

The selector's second member, P (the base map >= 0.5), has no GPU step: the CPU stage computes it from the base map.

Interpreters (environment variables; default: the current interpreter):
  FILAMENT_PY_CPU       requirements/cpu.txt                channels, selector stage
  FILAMENT_PY_NNUNET    requirements/nnunet-gpu.lock.txt    the two nnU-Nets
Weights (layout written by scripts/assemble_weights.py from weights/fast/):
  $FILAMENT_WEIGHTS/nnUNet_results/<Dataset>/nnUNetTrainer_250epochs__nnUNetPlans__2d/{plans.json,dataset.json,fold_0/}
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from filament_seg import filament_paths as fp

TRAINER_DIR = "nnUNetTrainer_250epochs__nnUNetPlans__2d"
SRC = Path(__file__).resolve().parents[2]
DS = {"D005": "Dataset005_FilamentFull", "D007": "Dataset007_FilamentVessel", "D036": "Dataset036_LimbDarkBarbClass"}
N_TEST = 180


def weights_nnunet() -> Path:
    return Path(os.environ.get("FILAMENT_WEIGHTS_NNUNET", fp.WEIGHTS / "nnUNet_results"))


def interpreter(kind: str) -> str:
    """The step's interpreter. A relative path is taken relative to the CALLER's directory, because every step runs
    with FILAMENT_WORK as its working directory (a bare name such as `python3` is left to PATH lookup)."""
    py = os.environ.get(f"FILAMENT_PY_{kind.upper()}", sys.executable)
    return os.path.abspath(py) if os.sep in py else py


@dataclass
class Step:
    """One resumable subprocess: `out` holds its outputs; it is complete when `count` files match `pattern`."""
    name: str
    env_kind: str                       # cpu | nnunet
    argv: list
    out: Path
    pattern: str
    count: int
    cwd: Path = field(default_factory=lambda: fp.WORK)
    extra_env: dict = field(default_factory=dict)
    pre: tuple = ()                     # (dataset, limit): fill that dataset's --limit subset before running
    required: tuple = ()                # if set: the step is complete iff every one of these files exists
    ok_marker: Path | None = None       # if set: also required, and written only after the step exits 0

    def done(self) -> bool:
        if self.ok_marker is not None and not self.ok_marker.exists():
            return False
        if self.required:
            return all(p.exists() for p in self.required)
        return self.out.is_dir() and len(list(self.out.glob(self.pattern))) >= self.count

    def missing(self) -> list:
        """The required files that do not exist yet (empty for a pattern-counted step)."""
        return [p for p in self.required if not p.exists()]


def env(extra: dict | None = None, limit: int = 0) -> dict:
    """The steps' environment: run-root paths, nnU-Net paths, PYTHONPATH; `extra` overrides."""
    return _env(extra or {}, limit)


def _env(extra: dict, limit: int = 0) -> dict:
    env = dict(os.environ)
    if limit:
        env["FILAMENT_LIMIT"] = str(limit)
    env.update(FILAMENT_WORK=str(fp.WORK), nnUNet_raw=str(fp.NNUNET_RAW), nnUNet_results=str(weights_nnunet()),
               nnUNet_preprocessed=str(fp.NNUNET_PREPROCESSED),
               nnUNet_compile="f", MPLBACKEND="Agg", PYTHONPATH=os.pathsep.join([str(SRC), env.get("PYTHONPATH", "")]).rstrip(os.pathsep))
    env.update(extra)
    return env


def _test_input(ds: str, limit: int) -> Path:
    """The imagesTs dir of a dataset, or (limit > 0) the dir that `_fill_subset` fills with its first `limit` cases."""
    full = fp.NNUNET_RAW / ds / "imagesTs"
    return full if not limit else fp.WORK / "limit" / f"{limit}" / ds / "imagesTs"


def _fill_subset(ds: str, limit: int) -> None:
    """Symlink the first `limit` test cases' channels of `ds` into its subset dir (idempotent; run just before use)."""
    if not limit:
        return
    full, sub = fp.NNUNET_RAW / ds / "imagesTs", _test_input(ds, limit)
    sub.mkdir(parents=True, exist_ok=True)
    cases = sorted({p.name[:-len("_0000.png")] for p in full.glob("*_0000.png")})[:limit]
    for c in cases:
        for f in full.glob(f"{c}_*.png"):
            if not (sub / f.name).exists():
                os.symlink(f.resolve(), sub / f.name)


def channel_files(datasets: list, limit: int) -> tuple:
    """Every channel file the slot's networks read: for each dataset, each test case (the first `limit` when limit > 0)
    times each of the dataset's channels (`configs/datasets/<Dataset>.dataset.json` channel_names)."""
    cases = sorted(json.loads((fp.CONFIGS / "case_mapping_full.json").read_text())["test"])[: limit or None]
    out = []
    for d in datasets:
        n_ch = len(json.loads((fp.CONFIGS / "datasets" / f"{DS[d]}.dataset.json").read_text())["channel_names"])
        out += [fp.NNUNET_RAW / DS[d] / "imagesTs" / f"{c}_{k:04d}.png" for c in cases for k in range(n_ch)]
    return tuple(out)


def _channels_step(datasets: list, limit: int) -> Step:
    """Complete only when EVERY dataset the slot needs has all its channels (a dataset built by an earlier, partial
    run does not make the step count as done)."""
    argv = ["-m", "filament_seg.prep.prepare", "--channels", "--split", "test", "--datasets", *datasets]
    if limit:
        argv += ["--limit", str(limit)]
    return Step("channels", "cpu", argv, fp.NNUNET_RAW, "", 0, required=channel_files(datasets, limit))


def _nnunet_step(name: str, member: str, folds: list, out: Path, limit: int, device: str,
                 probabilities: bool = False) -> Step:
    argv = ["-m", "filament_seg.gpu.nnunet_binaries", "--model-dir", str(weights_nnunet() / DS[member] / TRAINER_DIR),
            "--folds", *map(str, folds), "--in", str(_test_input(DS[member], limit)), "--out", str(out),
            "--step", "0.5", "--device", device]
    if probabilities:
        argv.append("--save-probabilities")
    return Step(name, "nnunet", argv, out, "*.png", limit or N_TEST, pre=(DS[member], limit))


def _selector_steps(slot: str, refit: bool, bag_dir: str | None) -> list:
    """The CPU stage. Default: APPLY the released selector bag (weights/selector/, sha256-checked); `bag_dir`: apply
    DIR/<slot>_bag.pkl; `refit` (full-retrain tier): build the selector tables, fit on the 707 out-of-fold rows (needs
    the out-of-fold bundle in WORK), then apply the fitted bag."""
    csv = str(fp.SUBMISSIONS_OUT / f"{slot}.csv")
    steps = []
    if refit:
        steps.append(Step("selector_tables", "cpu", ["-m", "filament_seg.selector_tables"], fp.WORK / "analysis",
                          "selector_tables.npz", 1))
        argv = ["-m", "filament_seg.selector_stage", "both", "--slot", slot, "--csv", csv]
    else:
        argv = ["-m", "filament_seg.selector_stage", "apply", "--slot", slot, "--csv", csv]
        if bag_dir:
            argv += ["--bag", str(Path(bag_dir).resolve() / f"{slot}_bag.pkl")]
    # The CSV is written BEFORE verify and the tolerance check run, so it exists even when the step fails; the marker
    # (written only on exit 0) keeps a re-run from treating a failed tolerance check as done.
    steps.append(Step(f"selector_{slot}", "cpu", argv, fp.SUBMISSIONS_OUT, f"{slot}.csv", 1,
                      ok_marker=fp.SUBMISSIONS_OUT / f".{slot}.ok"))
    return steps


def plan(slot: str, limit: int = 0, device: str = "cuda", refit: bool = False, bag_dir: str | None = None) -> list:
    """The ordered steps of one slot (only FAST, Ours-Fast, is predicted from this repository)."""
    P = fp.PREDICTIONS
    if slot == "FAST":
        steps = [_channels_step(["D007", "D036"], limit),          # the two networks' inputs only (D005 is not read by FAST)
                 _nnunet_step("base_D007_f0", "D007", [0], P / "vessel_full" / "prob", limit, device,
                              probabilities=True),
                 _nnunet_step("bin_D036_f0", "D036", [0], P / "test_d036_f0" / "by_image", limit, device)]
    else:
        raise ValueError(slot)
    if not limit:
        steps += _selector_steps(slot, refit, bag_dir)
    return steps


def run(slots: list, limit: int = 0, device: str = "cuda", dry_run: bool = False, refit: bool = False,
        bag_dir: str | None = None, log_dir: Path | None = None) -> int:
    """Execute (or, with dry_run, print) the steps of the slots in order; already-complete steps are skipped.
    With `log_dir`, each step's output goes to `<log_dir>/<step>.log` instead of the terminal."""
    seen, rc = set(), 0
    for slot in slots:
        for st in plan(slot, limit, device, refit=refit, bag_dir=bag_dir):
            key = (st.name, str(st.out), tuple(st.argv))      # argv: two channel steps differ by their datasets
            if key in seen:
                continue
            seen.add(key)
            line = {"slot": slot, "step": st.name, "env": st.env_kind, "python": interpreter(st.env_kind),
                    "argv": st.argv, "out": str(st.out), "complete": st.done()}
            print(json.dumps(line), flush=True)
            if dry_run or line["complete"]:
                continue
            st.out.mkdir(parents=True, exist_ok=True)
            if st.ok_marker is not None:
                st.ok_marker.unlink(missing_ok=True)
            if st.pre:
                _fill_subset(*st.pre)
            out = None
            if log_dir is not None:
                Path(log_dir).mkdir(parents=True, exist_ok=True)
                out = open(Path(log_dir) / f"{st.name}.log", "w")
            t0 = time.time()
            r = subprocess.run([interpreter(st.env_kind), *st.argv], cwd=str(st.cwd), env=_env(st.extra_env, limit),
                               stdout=out, stderr=subprocess.STDOUT if out else None)
            print(json.dumps({"step": st.name, "seconds": round(time.time() - t0, 1), "rc": r.returncode}), flush=True)
            if out:
                out.close()
            if r.returncode != 0:
                print(f"step {st.name} failed (rc {r.returncode}); re-run to resume"
                      + (f"; see {Path(log_dir) / (st.name + '.log')}" if log_dir is not None else ""), flush=True)
                return r.returncode
            if st.ok_marker is not None:
                st.ok_marker.write_text(json.dumps({"step": st.name, "argv": st.argv}) + "\n")
    return rc
