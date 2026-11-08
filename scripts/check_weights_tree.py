"""Check a local weights tree against weights/MANIFEST.sha256.

    python scripts/check_weights_tree.py ROOT [--manifest M]

ROOT is a tree in the `$FILAMENT_WEIGHTS` layout (`nnUNet_results/...`), for example the output of
`scripts/assemble_weights.py`. Passing means:
  * every model entry of the manifest (the networks of Ours-Fast) is present in ROOT with the manifest's sha256;
  * every other file in ROOT is a manifest entry with the manifest's sha256, apart from the names in `--ignore`
    (default: strip_report.json).
Exit 0 = pass, 1 = fail.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WEIGHTS = REPO / "weights"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def read_manifest(path: Path) -> dict:
    """layout path -> sha256, model entries only (the selector bags ship in the repository)."""
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            h, p = line.split(None, 1)
            if not p.strip().startswith("selector/"):
                out[p.strip()] = h
    return out


def check(root: Path, manifest: dict, ignore=("strip_report.json",)) -> dict:
    files = {str(p.relative_to(root)): p for p in sorted(root.rglob("*")) if p.is_file()}
    bad, unlisted, ok = [], [], {}
    missing = [rel for rel in manifest if rel not in files]
    for rel, p in files.items():
        if rel in ignore:
            continue
        if rel not in manifest:
            unlisted.append(rel)
            continue
        got = sha256(p)
        if got != manifest[rel]:
            bad.append(f"{rel}: sha256 {got}, manifest {manifest[rel]}")
        else:
            ok[rel] = (p.stat().st_size, got)
    return {"n_entries": len(manifest), "n_verified": len(ok), "missing": missing, "bad": bad, "unlisted": unlisted,
            "passed": not (missing or bad or unlisted), "verified": ok}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--manifest", type=Path, default=WEIGHTS / "MANIFEST.sha256")
    ap.add_argument("--ignore", nargs="*", default=["strip_report.json"])
    a = ap.parse_args(argv)
    manifest = read_manifest(a.manifest)
    res = check(a.root, manifest, tuple(a.ignore))
    for k in ("missing", "bad", "unlisted"):
        for x in res[k]:
            print(f"FAIL  {k}: {x}", file=sys.stderr)
    print(f"{res['n_verified']} files verified against the manifest ({res['n_entries']} entries); "
          f"{'PASS' if res['passed'] else 'FAIL'}")
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
