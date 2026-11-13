#!/usr/bin/env python3
"""Assemble the weights stored in the repository (`weights/nnHALO/`, the nnHALO networks) into $FILAMENT_WEIGHTS.

    python scripts/assemble_weights.py [--dest DIR]
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from filament_seg import weights  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", type=Path, help="target directory (default: $FILAMENT_WEIGHTS)")
    a = ap.parse_args()
    done = weights.assemble(a.dest)
    print(f"{len(done)} files in place and verified against weights/MANIFEST.sha256")
    return 0


if __name__ == "__main__":
    sys.exit(main())
