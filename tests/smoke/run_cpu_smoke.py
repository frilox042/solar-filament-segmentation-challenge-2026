"""CPU smoke test: this package rebuilds the uploaded nnHALO file BYTE-IDENTICALLY from cached network outputs.

It checks the following, each in a fresh `python -I` process (isolated: no PYTHONPATH,
no user site, no current directory on the path):

1. REFIT. `selector_stage both` fits the FLOOR bag on the 707 out-of-fold rows, then applies it to
   the 180 test images. The md5 must equal the uploaded file's.
2. FROZEN BAG. `selector_stage apply --bag <the pickle from 1>` runs in a new process. Same md5.
3. NEGATIVE CONTROL. Perturb one input of one test image (halve the D007 fold-0 test probabilities of that image)
   and apply the same bag. The md5 must DIFFER, which shows the check can fail.
2b. RELEASED BAG (the default predict path). `selector_stage apply` with no `--bag` applies
   `weights/selector/FAST_bag.pkl` (sha256-checked against weights/MANIFEST.sha256). Same md5. The
   refit's regressors and cut must also equal the released bag's (pickle bytes of the model list); as the negative
   control of that comparison, a copy of the released bag with one regressor attribute changed must NOT match.
4. ISOLATION. An audit hook in the launcher (inherited by forked workers) records every file opened
   for writing; none may resolve into SOURCE_ROOT. Every repo-local module in `sys.modules` must
   live inside this repository.

    python tests/smoke/run_cpu_smoke.py --python VENV/bin/python --source-root SRC --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SRC_PKG = REPO / "src"
sys.path.insert(0, str(HERE))
import stage_work  # noqa: E402

LAUNCH = r"""
import os, sys, json
REPO, SOURCE, WLOG = sys.argv[1], os.path.realpath(sys.argv[2]), sys.argv[3]
def _hook(ev, args):
    if ev == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)) and len(args) > 1:
        m = args[1]
        w = (isinstance(m, str) and any(c in m for c in "wax+")) or (isinstance(m, int) and m & (os.O_WRONLY | os.O_RDWR))
        if w:
            p = os.path.realpath(os.fsdecode(args[0]))
            if p.startswith(SOURCE + os.sep):
                with open(WLOG, "a") as f:   # nested open: not a write into SOURCE, no recursion issue
                    f.write(p + "\n")
sys.addaudithook(_hook)
sys.path.insert(0, os.path.join(REPO, "src"))
from filament_seg import selector_stage as S
rc = S.main(sys.argv[4:])
bad = []
for n, m in list(sys.modules.items()):
    f = getattr(m, "__file__", None) or ""
    if f and "site-packages" not in f and not f.startswith(sys.base_prefix) and not os.path.realpath(f).startswith(os.path.realpath(REPO)):
        bad.append((n, f))
print("FOREIGN_MODULES", json.dumps(bad))
sys.exit(rc)
"""


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def run(py, work, source, wlog, args, log):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("FILAMENT_", "nnUNet_", "PYTHON"))}
    env.update(FILAMENT_WORK=str(work), OMP_NUM_THREADS="4", CUDA_VISIBLE_DEVICES="")
    t0 = time.time()
    r = subprocess.run([py, "-I", "-c", LAUNCH, str(REPO), str(source), str(wlog), *args],
                       capture_output=True, text=True, env=env, cwd=str(work))
    open(log, "a").write(f"$ {' '.join(args)}\n{r.stdout}\n{r.stderr}\n")
    foreign = None
    for line in r.stdout.splitlines():
        if line.startswith("FOREIGN_MODULES"):
            foreign = json.loads(line.split(" ", 1)[1])
    return r.returncode, time.time() - t0, foreign


def perturb(slot, work, out):
    """Copy ONE input dir into the scratch area with one image changed; return (rel, real dir)."""
    import numpy as np
    rel = "predictions/vessel_full/prob"
    src = (work / rel).resolve()
    dst = out / f"nc_{slot}"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    files = sorted(p for p in src.iterdir() if p.suffix == ".npz")
    for p in src.iterdir():              # keep sidecar files (e.g. nnU-Net .pkl / json) visible as well
        if p not in files:
            os.symlink(p, dst / p.name)
    for p in files:
        os.symlink(p, dst / p.name)
    for p in files[8:]:          # first image with foreground that is not one of the 8 images the member-join gate
        z = dict(np.load(p))     # reads (or the gate refuses instead of the md5 changing): halve the D007 fold-0
        if (z["probabilities"][1] >= 0.97).sum() > 1000:          # probabilities, so the T97 mask shrinks
            (dst / p.name).unlink()
            z["probabilities"] = (z["probabilities"] * 0.5).astype(z["probabilities"].dtype)
            np.savez_compressed(dst / p.name, **z)
            return rel, dst, p.name
    raise SystemExit("no image to perturb")


def refit_and_frozen(a, slot, work, wlog, log, frozen):
    r = {}
    csv1 = a.out / f"{slot}_refit.csv"
    rc, s, fm = run(a.python, work, a.source_root, wlog, ["both", "--slot", slot, "--csv", str(csv1)], log)
    r["refit"] = {"rc": rc, "seconds": round(s), "md5": md5(csv1) if csv1.exists() else None, "foreign_modules": fm}
    shutil.copy(work / "selector_scratch" / slot / f"{slot}_bag.pkl", frozen)
    csv2 = a.out / f"{slot}_frozen_bag.csv"
    rc, s, fm = run(a.python, work, a.source_root, wlog, ["apply", "--slot", slot, "--csv", str(csv2), "--bag",
                                                           str(frozen)], log)
    r["frozen_bag"] = {"rc": rc, "seconds": round(s), "md5": md5(csv2) if csv2.exists() else None,
                       "foreign_modules": fm}
    csv4 = a.out / f"{slot}_released_bag.csv"
    rc, s, fm = run(a.python, work, a.source_root, wlog, ["apply", "--slot", slot, "--csv", str(csv4)], log)
    r["released_bag"] = {"rc": rc, "seconds": round(s), "md5": md5(csv4) if csv4.exists() else None,
                         "foreign_modules": fm,
                         "same_models_as_refit": same_models(frozen, released_bag(slot), a.python),
                         # negative control of that check: a released bag with one regressor attribute changed
                         "same_models_as_altered_bag": same_models(frozen, altered_bag(slot, a.out, a.python),
                                                                   a.python)}
    return r


def released_bag(slot):
    return REPO / "weights" / "selector" / f"{slot}_bag.pkl"


def altered_bag(slot, out, py) -> Path:
    """A copy of the released bag whose first regressor's learning_rate is changed (cut and the other models kept)."""
    dst = out / f"{slot}_bag_altered.pkl"
    code = ("import pickle, sys; d = pickle.load(open(sys.argv[1], 'rb')); m = d['bag'][0]; "
            "m.learning_rate = m.learning_rate * 1.5; pickle.dump(d, open(sys.argv[2], 'wb'))")
    subprocess.run([py, "-I", "-c", code, str(released_bag(slot)), str(dst)], check=True)
    return dst


def same_models(fitted, released, py) -> bool:
    """The two bags hold the same regressors and cut: same cut, same number of models, and every model's attribute
    dict equal key by key (each value compared by its pickle bytes). Whole-object pickles are NOT compared: two
    identical fits pickle to different bytes (object-sharing in the memo differs), seen on gate_p6 with identical
    attributes and identical predictions."""
    code = ("import pickle, sys; a = pickle.load(open(sys.argv[1], 'rb')); b = pickle.load(open(sys.argv[2], 'rb')); "
            "d = lambda m: {k: pickle.dumps(v, protocol=5) for k, v in vars(m).items()}; "
            "print(a['cut'] == b['cut'] and len(a['bag']) == len(b['bag']) "
            "and all(d(x) == d(y) for x, y in zip(a['bag'], b['bag'])))")
    r = subprocess.run([py, "-I", "-c", code, str(fitted), str(released)], capture_output=True, text=True)
    return r.stdout.strip() == "True"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--python", required=True)
    ap.add_argument("--source-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--slot", default="FAST", choices=("FAST",))
    ap.add_argument("--negctl-only", action="store_true", help="re-run only the negative control on an existing out dir")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    slots = [a.slot]
    res = {"python": a.python, "python_version": subprocess.run([a.python, "-c", "import sys;print(sys.version)"],
                                                                capture_output=True, text=True).stdout.strip(),
           "slots": {}}
    if a.negctl_only:
        res = json.load(open(a.out / "smoke_result.json"))
    for slot in slots:
        cfg = json.load(open(REPO / "configs" / f"{slot}.json"))
        work = a.out / f"work_{slot}"
        wlog, log = a.out / f"writes_into_source_{slot}.txt", a.out / f"log_{slot}.txt"
        frozen = a.out / f"{slot}_bag.pkl"
        if a.negctl_only:
            r = res["slots"][slot]
            r.setdefault("negative_control_earlier_attempts", []).append(r.get("negative_control"))
        else:
            if work.exists():
                shutil.rmtree(work)
            st = stage_work.stage(a.source_root, work, [slot])
            for p in (wlog, log):
                p.unlink(missing_ok=True)
            r = {"expected_md5": cfg["expected_md5"], "generated_inputs": st["generated"]}
            r.update(refit_and_frozen(a, slot, work, wlog, log, frozen))
        rel, ncdir, img = perturb(slot, work, a.out)
        link = work / rel
        orig = link.resolve()
        link.unlink()
        os.symlink(ncdir, link)
        csv3 = a.out / f"{slot}_negctl.csv"
        rc, s, fm = run(a.python, work, a.source_root, wlog, ["apply", "--slot", slot, "--csv", str(csv3), "--bag",
                                                               str(frozen)], log)
        link.unlink()
        os.symlink(orig, link)
        r["negative_control"] = {"perturbed": f"{rel}/{img}", "rc": rc, "seconds": round(s),
                                 "md5": md5(csv3) if csv3.exists() else None}
        writes = wlog.read_text().split() if wlog.exists() else []
        r["writes_into_source_root"] = writes
        rb = r.get("released_bag", {})
        r["PASS"] = (r["refit"]["md5"] == r["frozen_bag"]["md5"] == rb.get("md5") == cfg["expected_md5"]
                     and rb.get("same_models_as_refit") is True and rb.get("same_models_as_altered_bag") is False
                     and r["negative_control"]["md5"] not in (None, cfg["expected_md5"]) and not writes
                     and r["refit"]["foreign_modules"] == r["frozen_bag"]["foreign_modules"] == rb.get("foreign_modules")
                     == [])
        res["slots"][slot] = r
        print(slot, json.dumps(r), flush=True)
        json.dump(res, open(a.out / "smoke_result.json", "w"), indent=1)
    res["PASS"] = all(v["PASS"] for v in res["slots"].values())
    json.dump(res, open(a.out / "smoke_result.json", "w"), indent=1)
    print("PASS" if res["PASS"] else "FAIL")
    return 0 if res["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
