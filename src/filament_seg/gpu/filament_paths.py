"""Shim: the frozen modules `import filament_paths` as a top-level module. The one settings file is
`filament_seg/filament_paths.py`; this re-exports it so there is a single source of truth."""
from filament_seg.filament_paths import *  # noqa: F401,F403
from filament_seg.filament_paths import require  # noqa: F401
