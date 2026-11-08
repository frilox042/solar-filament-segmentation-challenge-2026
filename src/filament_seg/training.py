"""Training of nnHALO, as runnable, resumable steps: the full recipe of `docs/TRAINING.md`, and smoke runs of it.

    python scripts/train.py full        [--dry-run] [--root DIR]   # the recipe of the shipped weights (~65 GPU-h)
    python scripts/train.py smoke       [--dry-run] [--root DIR]   # 10 cases, fold 0, 1 epoch (~2.5 min): the notebook's check
    python scripts/train.py smoke-full  [--dry-run] [--root DIR]   # the FULL recipe on 10 cases, 5 folds x 1 epoch, every
                                                                   #   step up to the out-of-fold maps (the selector fit needs
                                                                   #   all 707 observations, so it is skipped)
    python scripts/train.py splits --dataset <Dataset> [--limit N] # write nnUNet_preprocessed/<Dataset>/splits_final.json

Training runs in its own run root (default `$FILAMENT_WORK/train`, or `--root`): its own nnUNet_raw / preprocessed /
results and predictions, so it never writes over the shipped weights or the prediction / evaluation inputs of the main
run root. The retrained networks are `<root>/nnunet_env/nnUNet_results` and the refitted selector bag is
`<root>/selector_scratch/FAST/FAST_bag.pkl`; `scripts/predict.py --from-images` uses them with
`FILAMENT_WEIGHTS_NNUNET=<root>/nnunet_env/nnUNet_results` and `--bag-dir <root>/selector_scratch/FAST`.
Every step is resumable: a step that completed writes `<root>/markers/<step>.ok` and is skipped on a re-run; nnU-Net
trainings continue from their last checkpoint (`--c`). nnU-Net training is NOT bit-reproducible on a GPU, so a retrain
gives different weights and a submission that differs slightly (docs/TRAINING.md).

Splits: identical for every dataset. Sort the case ids and take fold k = cases[k::5]; `splits_final.json[k]` =
{"train": every case not in fold k, "val": fold k}. On the 707 training cases this reproduces the shipped splits of
Dataset007 and Dataset036 exactly (tests/test_training.py checks the SHA-256 of the result).
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from filament_seg import filament_paths as fp
from filament_seg.gpu import pipeline as PL

REPO = Path(__file__).resolve().parents[2]
TRAINER = "nnUNetTrainer_250epochs"
SMOKE_TRAINER = "nnUNetTrainer_1epoch"
DATASETS = {7: "Dataset007_FilamentVessel", 36: "Dataset036_LimbDarkBarbClass"}
SMOKE_CASES = 10
SPLITS_SHA256 = "3b64bd504f44d1faf8b8c355d93dbfdb9149374ee219e2a9b2694b69eafa40b0"   # shipped D007 = D036 splits


def make_splits(cases: list, k: int = 5) -> list:
    cases = sorted(cases)
    folds = [cases[i::k] for i in range(k)]
    return [{"train": sorted(c for c in cases if c not in set(f)), "val": sorted(f)} for f in folds]


def splits_sha256(splits: list) -> str:
    return hashlib.sha256(json.dumps(splits, sort_keys=True).encode()).hexdigest()


def train_cases(limit: int = 0) -> list:
    return sorted(json.loads((fp.CONFIGS / "case_mapping_full.json").read_text())["train"])[: limit or None]


def write_splits(dataset: str, limit: int = 0) -> Path:
    """Write nnUNet_preprocessed/<dataset>/splits_final.json of the CURRENT run root (after plan_and_preprocess)."""
    out = fp.NNUNET_PREPROCESSED / dataset / "splits_final.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(make_splits(train_cases(limit)), indent=2))
    return out


def set_num_training(dataset: str, n: int) -> Path:
    """A limited (smoke) run trains on n cases: nnU-Net's integrity check needs dataset.json numTraining == n."""
    p = fp.NNUNET_RAW / dataset / "dataset.json"
    j = json.loads(p.read_text())
    j["numTraining"] = n
    p.write_text(json.dumps(j, indent=2))
    return p


def default_root() -> Path:
    return fp.WORK / "train"


def results_dir(root: Path | None = None) -> Path:
    """Where training writes its models (never the shipped weights)."""
    return Path(root or default_root()) / "nnunet_env" / "nnUNet_results"


def bag_dir(root: Path | None = None) -> Path:
    """Where the refitted selector bag is written (`--bag-dir` of scripts/predict.py)."""
    return Path(root or default_root()) / "selector_scratch" / "FAST"


@dataclass
class TStep:
    name: str
    kind: str        # cpu | nnunet
    argv: list       # arguments after the interpreter (run with cwd = the repository root)
    note: str = ""


PLAN = "nnunetv2.experiment_planning.plan_and_preprocess_entrypoints:plan_and_preprocess_entry"
TRAIN = "nnunetv2.run.run_training:run_training_entry"


def _nn(entry: str, args: list) -> list:
    """argv for an nnU-Net console entry point, run through the nnU-Net interpreter."""
    mod, fn = entry.split(":")
    return ["-c", f"import sys; sys.argv = {['nnunet'] + args!r}; from {mod} import {fn}; {fn}()"]


def _gpu_script(name: str) -> str:
    """The frozen out-of-fold tools import `filament_paths` as a top-level module (the shim next to them), so they
    run as scripts, not with -m."""
    return f"src/filament_seg/gpu/{name}.py"


def recipe(root: Path | None = None, limit: int = 0, trainer: str = TRAINER, fit: bool = True) -> list:
    """docs/TRAINING.md as steps. limit > 0: the first `limit` training cases only (a smoke run of the same commands)."""
    root = Path(root or default_root())
    lim = ["--limit", str(limit)] if limit else []
    res = results_dir(root)
    S = [TStep("channels", "cpu", ["scripts/prepare_data.py", "--channels", "--split", "train",
                                   "--datasets", "D007", "D036", *lim], "707 training images, several in parallel"),
         TStep("labels", "cpu", ["scripts/build_labels.py", "--datasets", "D007", "D036", *lim],
               "union label (D007), barb-class label (D036), from the competition COCO")]
    if fit:
        S.append(TStep("selector tables", "cpu", ["scripts/prepare_data.py", "--selector-tables"]))
    for i, ds in DATASETS.items():
        if limit:
            S.append(TStep(f"numTraining {ds}", "cpu", ["scripts/train.py", "set-num-training", "--dataset", ds,
                                                        "--n", str(limit)]))
        S.append(TStep(f"plan + preprocess {ds}", "nnunet", _nn(PLAN, ["-d", str(i), "-c", "2d",
                                                                        "--verify_dataset_integrity"])))
        S.append(TStep(f"splits {ds}", "cpu", ["scripts/train.py", "splits", "--dataset", ds, *lim]))
        for f in range(5):
            S.append(TStep(f"train {ds} fold {f}", "nnunet", _nn(TRAIN, [str(i), "2d", str(f), "-tr", trainer, "--c"]),
                           "~6.5 h on the reference laptop GPU" if trainer == TRAINER else "1 epoch (smoke)"))
    tdir = f"{trainer}__nnUNetPlans__2d"
    S.append(TStep("out-of-fold probability maps (D007)", "nnunet",
                   [_gpu_script("predict_oof_probs"), "--out", str(root / "predictions" / "vessel_oof_probs"),
                    "--step-size", "0.75", "--folds", "0", "1", "2", "3", "4", "--trainer-dir", tdir, "--resume"]))
    d36 = res / DATASETS[36] / tdir
    for f in range(5):
        S.append(TStep(f"D036 fold {f} validation -> binary", "cpu",
                       [_gpu_script("merge_barb_predictions"), str(d36 / f"fold_{f}" / "validation")]))
    S.append(TStep("out-of-fold masks (D036)", "cpu",
                   [_gpu_script("make_oof_set"), "--dataset", DATASETS[36], "--trainer", tdir,
                    "--out", str(root / "predictions" / "oof_d036" / "by_image"),
                    "--from-dirs", str(d36 / "fold_%d" / "validation_binary")] + (["--allow-partial"] if limit else [])))
    if fit:
        S.append(TStep("selector fit", "cpu", ["-m", "filament_seg.selector_stage", "fit", "--retrain"],
                       "~5.5 min CPU; writes " + str(bag_dir(root) / "FAST_bag.pkl")))
    return S


def full_recipe(root: Path | None = None) -> list:
    return recipe(root)


def smoke_full_recipe(root: Path | None = None) -> list:
    """The full recipe's commands on the first 10 training cases, 5 folds x 1 epoch, every step up to the out-of-fold
    maps; the selector fit (all 707 observations) is skipped."""
    return recipe(root, limit=SMOKE_CASES, trainer=SMOKE_TRAINER, fit=False)


def smoke_recipe(root: Path | None = None) -> list:
    """The notebook's quick check: 10 cases of Dataset007, plan + preprocess, splits, 1 epoch of fold 0."""
    lim = ["--limit", str(SMOKE_CASES)]
    ds = DATASETS[7]
    return [TStep("channels", "cpu", ["scripts/prepare_data.py", "--channels", "--split", "train", "--datasets", "D007",
                                      *lim]),
            TStep("labels", "cpu", ["scripts/build_labels.py", "--datasets", "D007", *lim]),
            TStep(f"numTraining {ds}", "cpu", ["scripts/train.py", "set-num-training", "--dataset", ds,
                                               "--n", str(SMOKE_CASES)]),
            TStep(f"plan + preprocess {ds}", "nnunet", _nn(PLAN, ["-d", "7", "-c", "2d", "--verify_dataset_integrity"])),
            TStep(f"splits {ds}", "cpu", ["scripts/train.py", "splits", "--dataset", ds, *lim]),
            TStep(f"train {ds} fold 0", "nnunet", _nn(TRAIN, ["7", "2d", "0", "-tr", SMOKE_TRAINER, "--c"]),
                  "1 epoch (smoke)")]


def env(root: Path) -> dict:
    """The steps' environment: the prediction pipeline's, with every run-root path moved to `root`."""
    root = Path(root)
    nn = root / "nnunet_env"
    return PL.env({"FILAMENT_WORK": str(root), "nnUNet_raw": str(nn / "nnUNet_raw"),
                   "nnUNet_preprocessed": str(nn / "nnUNet_preprocessed"), "nnUNet_results": str(nn / "nnUNet_results")})


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def run(steps: list, root: Path | None = None, dry_run: bool = False, quiet: bool = False) -> int:
    """Run steps in order with each kind's interpreter (cwd: the repository root). A completed step leaves
    `<root>/markers/<step>.ok` and is skipped on a re-run. Logs: `<root>/logs/<step>.log`. quiet: print one line per
    step (name and outcome) instead of the full command (which contains absolute interpreter paths)."""
    root = Path(root or default_root())
    (root / "markers").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    e = env(root)
    for st in steps:
        mark = root / "markers" / f"{_slug(st.name)}.ok"
        if quiet:
            print(f"  {st.name:44s} " + ("listed" if dry_run else "already done" if mark.exists() else "running"),
                  flush=True)
        else:
            print(json.dumps({"step": st.name, "kind": st.kind, "python": PL.interpreter(st.kind), "argv": st.argv,
                              "note": st.note, "done": mark.exists()}), flush=True)
        if dry_run or mark.exists():
            continue
        with open(root / "logs" / f"{_slug(st.name)}.log", "w") as out:
            r = subprocess.run([PL.interpreter(st.kind), *st.argv], cwd=str(REPO), env=e, stdout=out,
                               stderr=subprocess.STDOUT)
        if r.returncode != 0:
            print(f"step {st.name!r} failed (rc {r.returncode}); log: logs/{_slug(st.name)}.log under the run root",
                  flush=True)
            return r.returncode
        mark.write_text(json.dumps({"step": st.name, "argv": st.argv}) + "\n")
    return 0
