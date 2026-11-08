"""CPU stage: network outputs -> base mask -> per-object FLOOR selector -> CSV.

    python -m filament_seg.selector_stage apply --slot FAST [--bag PKL] [--csv OUT]
    python -m filament_seg.selector_stage both --slot FAST [--csv OUT]
    python -m filament_seg.selector_stage fit --slot FAST

`apply` (the default predict path) renders the 180 test images with the RELEASED selector bag,
`$FILAMENT_SELECTOR_BAGS/FAST_bag.pkl` (shipped in `weights/selector/`). Its sha256 is checked against
`weights/MANIFEST.sha256` before it is unpickled. `--bag PKL` applies another bag instead, for example one written by
`fit`. `fit` (the full-retrain tier) trains the selector on the 707 out-of-fold training rows and writes a bag pickle
to the slot's scratch dir. That needs the out-of-fold bundle and `analysis/selector_tables.npz`. `both` fits, then
applies THAT freshly fitted bag, never the released one.

Exit status. After `apply` the CSV is compared with the uploaded file `submissions/FAST.csv` under the
`reproduction_tolerance` rule of `configs/FAST.json` (`filament_seg.submission.compare`): at most 3 images whose object
count differs, each by at most 1 object, total object count within +-3, and pooled union-mask Dice over the images
whose count matches >= 0.999.
The status is 0 when `verify_submission` passes and the rule passes (and, after a fit, the fitted cut equals the
config's). Reported as information only: `md5_matches_reference` (bit-identical) and `same_count_everywhere`. The md5
match does not decide the status: the GPU stage is non-deterministic, so a from-images run is
not expected to match the uploaded file byte for byte.

All settings come from `configs/<slot>.json` (`filament_seg.config`); all inputs are read from the run root
`FILAMENT_WORK` (layout: docs/work_layout.md). Nothing here changes the working directory or any module-level state.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

from filament_seg import config as C
from filament_seg import filament_paths as fp


def bag_path(cfg: C.SlotConfig) -> str:
    """Where `fit` writes the slot's bag (the scratch dir of the run root)."""
    return str(cfg.scratch / f"{cfg.slot}_bag.pkl")


def released_bag_path(slot: str) -> Path:
    """The released (deployed) selector bag of a slot."""
    return fp.SELECTOR_BAGS / f"{slot}_bag.pkl"


def manifest_sha256(rel: str, manifest: Path | None = None) -> str:
    """The sha256 that `weights/MANIFEST.sha256` records for a layout path such as `selector/FAST_bag.pkl`."""
    manifest = Path(manifest or fp.WEIGHTS_MANIFEST)
    for line in manifest.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2 and not line.startswith("#") and parts[1] == rel:
            return parts[0]
    raise KeyError(f"{rel} is not listed in {manifest}")


def load_released_bag(slot: str) -> dict:
    """Unpickle the released bag of `slot` ONLY after its sha256 matches the manifest (integrity, and pickle safety:
    a file that is not the released one is never executed)."""
    p = released_bag_path(slot)
    data = fp.require(p, f"released {slot} selector bag", "it ships in weights/selector/").read_bytes()
    want, got = manifest_sha256(f"selector/{slot}_bag.pkl"), hashlib.sha256(data).hexdigest()
    if got != want:
        raise ValueError(f"refusing to load {p}: sha256 {got} differs from weights/MANIFEST.sha256 ({want})")
    return pickle.loads(data)


def run_fit(cfg: C.SlotConfig) -> dict:
    """Fit the slot's selector and write its bag pickle; returns the fit summary."""
    from filament_seg import selector_fit
    bag, cut, info = selector_fit.fit(cfg)
    info.update(slot=cfg.slot, workers=cfg.fit_workers, two_cut_matches_config=info["two_cut"] == cfg.expected_two_cut)
    os.makedirs(cfg.scratch, exist_ok=True)
    pickle.dump({"bag": bag, "cut": cut, "info": info}, open(bag_path(cfg), "wb"))
    return info


def run_apply(cfg: C.SlotConfig, csv_path: str, bag: str | None = None) -> dict:
    """Apply a bag to the test images and compare the CSV with the uploaded file.

    `bag`: a pickle path, or None for the released bag (sha256-checked)."""
    from filament_seg import render
    from filament_seg.submission import compare
    d = load_released_bag(cfg.slot) if bag is None else pickle.load(open(bag, "rb"))
    t0 = time.time()
    res = render.apply(cfg, d["bag"], d["cut"], csv_path, verify_json=cfg.scratch / "verify.json")
    cmp = compare.compare(csv_path, cfg.reference_csv, C.test_image_ids(cfg), cfg.tolerance_rule)
    tol = {k: cmp[k] for k in ("reference", "reference_rows", "images_count_differs", "pooled_dice_count_matched",
                               "pooled_dice_all", "min_matched_image_dice", "worst_matched_image", "min_image_dice",
                               "worst_image", "images_identical_rows", "failures")}
    return {"slot": cfg.slot, "csv": os.path.abspath(csv_path),
            "bag": "released" if bag is None else os.path.abspath(bag),
            "md5": cmp["md5"], "expected_md5": cfg.expected_md5, "md5_matches_reference": cmp["md5_matches_reference"],
            "rows": res["n_rows"], "verify_rc": res["verify_rc"], "within_tolerance": cmp["within_tolerance"],
            "same_count_everywhere": cmp["same_count_everywhere"],
            "tolerance": tol, "apply_s": time.time() - t0}


def status(out: dict, retrain: bool = False) -> int:
    """0 when every step that ran passed: fit cut == config (not required with retrain: retrained networks give a
    different cut), verify rc 0, CSV within the reproduction rule (not required with retrain). md5 is not checked."""
    ok = True if retrain else out.get("fit", {}).get("two_cut_matches_config", True)
    if "apply" in out:
        ok = ok and out["apply"]["verify_rc"] == 0 and (retrain or out["apply"]["within_tolerance"])
    return 0 if ok else 1


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("fit", "apply", "both"))
    ap.add_argument("--slot", default="FAST", choices=("FAST",), help="the configuration (configs/FAST.json)")
    ap.add_argument("--csv", help="output CSV (default submissions/out/FAST.csv)")
    ap.add_argument("--bag", help="apply: a bag pickle to apply instead of the released one (e.g. written by `fit`)")
    ap.add_argument("--retrain", action="store_true",
                    help="the members were retrained: the cut and the CSV are reported, not gated against the released ones")
    a = ap.parse_args(argv)
    if a.cmd == "both" and a.bag:
        ap.error("`both` applies the bag it fits; --bag is for `apply`")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    cfg = C.load(a.slot)
    out = {}
    if a.cmd in ("fit", "both"):
        out["fit"] = run_fit(cfg)
        print(json.dumps(out["fit"]), flush=True)
    if a.cmd in ("apply", "both"):
        bag = bag_path(cfg) if a.cmd == "both" else a.bag
        out["apply"] = run_apply(cfg, a.csv or str(fp.SUBMISSIONS_OUT / f"{a.slot}.csv"), bag)
        print(json.dumps(out["apply"]), flush=True)
    return status(out, a.retrain)


if __name__ == "__main__":
    sys.exit(main())
