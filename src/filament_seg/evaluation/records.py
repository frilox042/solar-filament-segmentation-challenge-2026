"""Which per-annotator-record source an evaluation reads: `coco` (default) or `png` (historical replication only).

    src = records.source("coco")      # module with records_by_case(train_case_map), record_mask(r), union_mask(rs)

coco  `records_coco`: the competition training COCO, the only label file the competition ships (reviewer-runnable).
png   `records_png`: the `extracted/training` PNG tree, a PNG export of the same records. The reported
      annotator-agreement and never-seen numbers were measured on it; `--records png` reproduces them, nothing else.
"""
from __future__ import annotations

from types import ModuleType

SOURCES = ("coco", "png")
DEFAULT = "coco"


def source(name: str = DEFAULT) -> ModuleType:
    if name == "coco":
        from filament_seg.evaluation import records_coco
        return records_coco
    if name == "png":
        from filament_seg.evaluation import records_png
        return records_png
    raise ValueError(f"unknown records source {name!r}; one of {SOURCES}")


def add_argument(ap) -> None:
    """The scripts' shared `--records` flag."""
    ap.add_argument("--records", choices=SOURCES, default=DEFAULT,
                    help="per-record annotation source: coco (default, the competition COCO) or png (the historical "
                         "extracted/ PNG tree; only to replicate the published values)")
