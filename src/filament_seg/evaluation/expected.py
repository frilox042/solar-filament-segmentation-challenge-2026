"""The published values each evaluation gate must reproduce (`configs/eval/expected.json`), and the gate check."""
from __future__ import annotations

import json

import numpy as np

from filament_seg import filament_paths as fp


def load(section: str, model_set: str | None = None, records: str | None = None) -> dict:
    """One section (e.g. "heldout_annotator") of the expected values of a model set. Default: the block of the
    per-record source `records` (`by_records` in expected.json: png = the reported values, coco = this
    code's own COCO run), else the current one."""
    j = json.loads((fp.CONFIGS / "eval" / "expected.json").read_text())
    name = model_set or (j["by_records"][records] if records else j["current"])
    if section not in j[name]:
        raise KeyError(f"configs/eval/expected.json block {name!r} has no section {section!r}")
    return j[name][section]


def close(mine, published, tol: float) -> dict:
    """{mine, published, max_abs, pass}: element-wise |mine - published| <= tol (nested lists allowed)."""
    a, b = np.asarray(mine, float), np.asarray(published, float)
    d = float(np.max(np.abs(a - b))) if a.shape == b.shape else float("inf")
    return {"mine": np.asarray(mine).tolist(), "published": np.asarray(published).tolist(), "max_abs": d,
            "pass": bool(d <= tol)}


def compare_tree(mine, published, tol: float, path: str = "") -> dict:
    """{dotted path: close(...)} for every numeric leaf of `published` (keys missing from `mine` fail)."""
    out = {}
    if isinstance(published, dict):
        for k, v in published.items():
            sub = mine.get(k) if isinstance(mine, dict) else None
            out.update(compare_tree(sub, v, tol, f"{path}.{k}" if path else k))
        return out
    if mine is None:
        return {path: {"mine": None, "published": published, "pass": False}}
    return {path: close(mine, published, tol)}
