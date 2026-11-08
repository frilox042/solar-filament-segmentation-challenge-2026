"""Merge a 3-class (background / filament / barb) prediction directory into a binary one.

⛔ WHY THIS IS EXPLICIT RATHER THAN IMPLICIT. `score_arm_pq.py` reads `>0` as foreground, so a
3-class prediction would already score its barb class as foreground -- but only by accident of that
convention. A future change to either side would silently alter what the arm means. The barb class
is SUPERVISION, not a scored output, so the merge is made explicit and the arm is measured against
exactly the binary reference every other arm uses.

Usage: python merge_barb_predictions.py <fold_dir>/validation [--expect N]
Writes <fold_dir>/validation_binary. The expected prediction count is that fold's own validation
split size, read from the dataset's splits_final.json; --expect overrides it.
"""
import argparse, glob, json, os, re, shutil, sys
import numpy as np
from PIL import Image

from filament_paths import NNUNET_PREPROCESSED


def expected_count(src):
    """The validation-split size of the fold `src` belongs to, or None if the path does not name one.

    The fold sizes differ (707 = 142+142+141+141+141), so the count cannot be a constant: it comes from the split the
    fold was trained on, which is the only source that knows the right number.
    """
    m = re.search(r"/(Dataset\d+_[^/]+)/[^/]+/fold_(\d+)[^/]*/validation$",
                  os.path.abspath(src).rstrip("/"))
    if not m:
        return None
    ds, k = m.group(1), int(m.group(2))
    p = NNUNET_PREPROCESSED / ds / "splits_final.json"
    if not p.exists():
        return None
    s = json.loads(p.read_text())
    return len(s[k]["val"]) if k < len(s) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--expect", type=int, default=None,
                    help="expected prediction count (default: the fold's validation-split size)")
    a = ap.parse_args()
    src = a.src.rstrip("/")
    dst = src + "_binary"
    tmp = dst + ".partial"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    n = 0
    seen = set()
    for f in sorted(glob.glob(os.path.join(src, "*.png"))):
        arr = np.asarray(Image.open(f))
        if arr.ndim == 3:
            arr = arr[..., 0]
        seen.update(np.unique(arr).tolist())
        Image.fromarray((arr > 0).astype(np.uint8), mode="L").save(os.path.join(tmp, os.path.basename(f)))
        n += 1
    print(f"  merged {n} predictions -> {dst}; source label values seen: {sorted(seen)}")

    # ⛔ AN INFRASTRUCTURE FAILURE MUST NOT BE READABLE AS A SCIENTIFIC VERDICT. Run on a nonexistent
    # directory the previous version printed "no class-2 pixels anywhere -- read that as the arm's
    # result, not as a merge failure" and exited 0. That is the inversion of the defect score_arm_pq.py
    # was built to prevent: zero inputs instructing the reader to treat the emptiness as the finding.
    # And a refused merge must leave NO output behind: a chain that skips the merge when the output
    # directory exists would otherwise accept a partial set on its next pass.
    if n == 0:
        shutil.rmtree(tmp, ignore_errors=True)
        sys.exit(f"REFUSING: {src} contained no predictions. This is a MERGE FAILURE, not an arm result.")
    exp = a.expect if a.expect is not None else expected_count(src)
    how = "--expect" if a.expect is not None else "this fold's validation split"
    if exp is None:
        exp, how = 142, "FALLBACK -- the path names no fold of a dataset with a splits_final.json, so fold 0's 142 is assumed"
    if n != exp:
        shutil.rmtree(tmp, ignore_errors=True)
        sys.exit(f"REFUSING: merged {n} predictions, expected {exp} ({how}). A partial validation set "
                 f"scores a different population and would enter a seed mean as a peer. Nothing written.")
    if 2 not in seen:
        # now a genuine finding rather than a cover for an empty directory, because n == exp above
        print(f"  NOTE: {n} predictions merged but NO class-2 pixels anywhere. The barb class was never "
              "predicted, which IS the arm's result and should be reported as such.")
    shutil.rmtree(dst, ignore_errors=True)
    os.rename(tmp, dst)
    print(f"  OK: {n} == {exp} ({how})")


if __name__ == "__main__":
    main()
