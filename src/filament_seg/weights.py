"""Assemble the weights stored in this repository into the `$FILAMENT_WEIGHTS` layout.

`weights/fast/` holds the two networks of Ours-Fast in plain git: GitHub blocks files above 100 MiB, so each
checkpoint is stored as `<name>.part000`, `.part001`, ... of at most 90 MiB. `assemble` concatenates the parts (small files are copied as they are) and checks every resulting file
against `weights/MANIFEST.sha256`; a file whose hash does not match is deleted and the call fails. Files that are
already in place with the right hash are skipped, so it is safe to call repeatedly.

    python scripts/assemble_weights.py            # -> $FILAMENT_WEIGHTS (default: weights/downloaded)
"""
from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path

from filament_seg import filament_paths as fp

REPO_WEIGHTS = fp.REPO / "weights"
PARTS_DIR = REPO_WEIGHTS / "fast"
PART_RE = re.compile(r"^(?P<name>.+)\.part(?P<i>\d{3})$")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def manifest() -> dict:
    """layout path -> sha256 for every entry of `weights/MANIFEST.sha256`."""
    out = {}
    for line in (REPO_WEIGHTS / "MANIFEST.sha256").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            digest, path = line.split(None, 1)
            out[path.strip()] = digest
    return out


def stored_files(parts_dir: Path = PARTS_DIR) -> dict:
    """layout path -> list of source files (one file, or the ordered parts) under `parts_dir`."""
    found: dict = {}
    for p in sorted(parts_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(parts_dir).as_posix()
        m = PART_RE.match(rel)
        found.setdefault(m["name"] if m else rel, []).append(p)
    return found


def assemble(dest: Path | None = None, parts_dirs=(PARTS_DIR,), expected: dict | None = None) -> list:
    """Write every stored file into `dest` (default `$FILAMENT_WEIGHTS`), verified against the manifest.
    Returns the layout paths now in place. Raises if a stored file is not in the manifest or a hash differs."""
    dest = Path(dest) if dest is not None else fp.WEIGHTS
    expected = manifest() if expected is None else expected
    if isinstance(parts_dirs, (str, Path)):
        parts_dirs = (parts_dirs,)
    done = []
    stored = {}
    for d in parts_dirs:
        if Path(d).is_dir():
            stored.update(stored_files(Path(d)))
    for rel, sources in stored.items():
        if rel not in expected:
            raise ValueError(f"{rel} is stored in the repository but is not in MANIFEST.sha256")
        target = dest / rel
        if target.exists() and sha256(target) == expected[rel]:
            done.append(rel)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        with open(tmp, "wb") as out:
            for s in sources:
                with open(s, "rb") as f:
                    shutil.copyfileobj(f, out, 1 << 22)
        if sha256(tmp) != expected[rel]:
            tmp.unlink()
            raise ValueError(f"{rel}: sha256 does not match MANIFEST.sha256")
        tmp.replace(target)
        done.append(rel)
    return done
