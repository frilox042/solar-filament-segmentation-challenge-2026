"""Assemble a run root (WORK) for the CPU smoke test from an existing set of member maps.

The inputs are exactly those in `work_manifest.json`, which was recorded by tracing the
files the stage opens. Each per-image directory and each single file is SYMLINKED read-only from the
source tree. A directory that receives output is never linked: `analysis/` and `predictions/` are real
directories inside WORK, and only their leaves are links.

`file_as` entries map a WORK path to a different source path: the competition COCO is staged at its default
location in the competition download layout (`data/kaggle/MAGFiLO_1.0_Kaggle_2026/train/...`, see
`filament_paths.KAGGLE_COCO`).

Generated inputs (`generate` in the manifest) are BUILT inside WORK by this repository's own code, not linked:
`analysis/selector_tables.npz` (case list + per-record COCO counts) from the competition COCO and the case
mapping.

    python tests/smoke/stage_work.py --source-root SRC --work WORK
    python tests/smoke/stage_work.py --source-root SRC --work WORK --eval    # offline evaluation inputs only

SRC is a tree laid out like WORK, holding the out-of-fold maps and the test network outputs.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _link(src: Path, dst: Path):
    if not src.exists():
        raise SystemExit(f"missing input: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() and dst.resolve() == src.resolve():
            return
        raise SystemExit(f"refusing to replace existing {dst}")
    os.symlink(src.resolve(), dst)


def _generate(rel: str, source_root: Path, work: Path) -> dict:
    """Build a generated input inside WORK with this repository's own code (in a child process, cwd/env = WORK)."""
    import subprocess
    import sys
    assert rel == "analysis/selector_tables.npz", rel
    code = ("import sys, json; sys.path.insert(0, %r); from filament_seg import selector_tables as T; "
            "print(T.write())" % str(HERE.parents[1] / "src"))
    env = dict(os.environ, FILAMENT_WORK=str(work))
    env.pop("nnUNet_raw", None)
    env.pop("FILAMENT_KAGGLE_COCO", None)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=str(work))
    if r.returncode != 0:
        raise SystemExit(f"generate {rel} failed:\n{r.stderr}")
    return {"path": r.stdout.strip().splitlines()[0]}


def stage(source_root: Path, work: Path, slots: list[str]) -> dict:
    man = json.loads((HERE / "work_manifest.json").read_text())
    work.mkdir(parents=True, exist_ok=True)
    groups = ["common", "fit_common"] + slots
    n = 0
    out = {"generated": {}}
    for g in groups:
        for rel in man[g].get("file", []) + man[g].get("dir", []):
            _link(source_root / rel, work / rel)
            n += 1
        for rel, src_rel in man[g].get("file_as", {}).items():     # staged at a different path than in the source
            _link(source_root / src_rel, work / rel)
            n += 1
        for rel in man[g].get("generate", {}):
            out["generated"][rel] = _generate(rel, source_root, work)
    return {"work": str(work), "links": n, "slots": slots, **out}


def stage_eval(source_root: Path, work: Path) -> dict:
    """The offline evaluation's inputs (`configs/eval/inputs.json`): OOF maps and binaries, competition data."""
    man = json.loads((HERE.parents[1] / "configs" / "eval" / "inputs.json").read_text())
    work.mkdir(parents=True, exist_ok=True)
    for rel in man["file"] + man["dir"]:
        _link(source_root / rel, work / rel)
    for rel, src_rel in man.get("file_as", {}).items():
        _link(source_root / src_rel, work / rel)
    # optional inputs (the historical PNG tree, only for --records png): staged when the source has them
    opt = {rel: (source_root / rel).exists() for rel in man.get("optional_dir", [])}
    for rel, ok in opt.items():
        if ok:
            _link(source_root / rel, work / rel)
    gen = {rel: _generate(rel, source_root, work) for rel in man["generate"]}
    n_links = len(man["file"]) + len(man["dir"]) + len(man.get("file_as", {})) + sum(opt.values())
    return {"work": str(work), "links": n_links, "eval": True,
            "optional_staged": opt, "generated": gen}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-root", required=True, type=Path)
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--slot", default="FAST", choices=("FAST",))
    ap.add_argument("--fresh", action="store_true", help="delete WORK first")
    ap.add_argument("--eval", action="store_true", help="stage the offline evaluation inputs instead of a slot")
    a = ap.parse_args()
    if a.fresh and a.work.exists():
        shutil.rmtree(a.work)
    if a.eval:
        print(json.dumps(stage_eval(a.source_root, a.work)))
        return
    print(json.dumps(stage(a.source_root, a.work, [a.slot])))


if __name__ == "__main__":
    main()
