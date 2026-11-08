#!/usr/bin/env python3
"""Data counts quoted in the report, recomputed from the competition files (training COCO + case mapping).

    FILAMENT_KAGGLE_COCO=... python scripts/report_data_numbers.py
Prints: training observations, annotator records per observation (1/2/3), observations with >= 2 records, and the
held-out-record units of the agreement evaluation (sum of records over multi-record observations).
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np  # noqa: E402

from filament_seg import selector_tables  # noqa: E402


def main() -> int:
    t = selector_tables.build()
    k = t["coco_k"]
    multi = k >= 2
    out = {"training_observations": int(len(k)),
           "records_per_observation": {str(v): int((k == v).sum()) for v in np.unique(k)},
           "observations_with_2_or_3_records": int(multi.sum()),
           "held_out_record_units": int(k[multi].sum()),
           "annotated_filaments_total": int(t["coco_G"].sum())}
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
