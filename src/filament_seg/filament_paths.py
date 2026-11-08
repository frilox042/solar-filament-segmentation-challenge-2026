"""THE settings file (the Kaggle ``SETTINGS.json`` equivalent): every filesystem location used by any stage.

Resolution order for every path: environment variable, then a default relative to this repository.
No other module may hold a path of its own, except the standalone GPU scripts in ``gpu/``. Those
use paths RELATIVE TO THE RUN ROOT (``WORK``), for example ``predictions/...`` and ``analysis/...``,
and they ``chdir`` into ``WORK`` when imported. The defaults below therefore place every tree under
``WORK`` so the two views agree.

Data roles. These are kept explicit because the competition rules depend on them.

* ``KAGGLE_ROOT`` is the unpacked competition download: train images, the training COCO JSON, and the
  test images.
* ``KAGGLE_COCO`` is the competition's own training annotation file. It is byte-identical to
  ``MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json`` in
  ``filament-segmentation-2026.zip`` (md5 ec641102e51c340ef9debbeb1fd5e663).
* ``MAGFILO_JSON`` is the public MAGFiLO 1.0 release. It contains the TEST annotations. **No stage of
  this repository reads it.** It is defined only so that ``tests/test_no_magfilo.py`` can point it at a
  nonexistent file and prove that every stage still runs.

=======================  ===================================================================
FILAMENT_REPO            repository root (default: two levels above this file)
FILAMENT_WORK            run root: caches, member maps, outputs (default: $FILAMENT_REPO/work)
FILAMENT_DATA            data root (default: $FILAMENT_WORK/data)
FILAMENT_KAGGLE_ROOT     unpacked competition download (default: $FILAMENT_DATA/kaggle)
FILAMENT_KAGGLE_COCO     competition training COCO JSON (default: the download layout,
                         $FILAMENT_KAGGLE_ROOT/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json)
FILAMENT_EXTRACTED       per-annotator-record PNG tree (training/<rec>/...). NOT competition data: a PNG export. Read only by the historical `--records png` replication.
FILAMENT_WEIGHTS         released weights (default: $FILAMENT_REPO/weights/downloaded)
FILAMENT_SELECTOR_BAGS   released selector bags <slot>_bag.pkl (default: $FILAMENT_REPO/weights/selector, in the repo)
nnUNet_raw / nnUNet_preprocessed / nnUNet_results   nnU-Net's own variables (default: $FILAMENT_WORK/nnunet_env/...)
=======================  ===================================================================
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(os.environ.get("FILAMENT_REPO", Path(__file__).resolve().parents[2]))
WORK = Path(os.environ.get("FILAMENT_WORK", REPO / "work"))
DATA = Path(os.environ.get("FILAMENT_DATA", WORK / "data"))

KAGGLE_ROOT = Path(os.environ.get("FILAMENT_KAGGLE_ROOT", DATA / "kaggle"))
# The competition download layout: <KAGGLE_ROOT>/MAGFiLO_1.0_Kaggle_2026/train/<this file>.
KAGGLE_COCO = Path(os.environ.get(
    "FILAMENT_KAGGLE_COCO",
    KAGGLE_ROOT / "MAGFiLO_1.0_Kaggle_2026" / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
EXTRACTED = Path(os.environ.get("FILAMENT_EXTRACTED", WORK / "extracted"))
# Never read by any stage; see the module docstring.
MAGFILO_JSON = Path(os.environ.get("FILAMENT_MAGFILO_JSON", DATA / "magfilo_full.json.NOT_USED"))

NNUNET_RAW = Path(os.environ.get("nnUNet_raw", WORK / "nnunet_env" / "nnUNet_raw"))
NNUNET_PREPROCESSED = Path(os.environ.get("nnUNet_preprocessed", WORK / "nnunet_env" / "nnUNet_preprocessed"))
NNUNET_RESULTS = Path(os.environ.get("nnUNet_results", WORK / "nnunet_env" / "nnUNet_results"))

WEIGHTS = Path(os.environ.get("FILAMENT_WEIGHTS", REPO / "weights" / "downloaded"))
# The released (deployed) selector bags, <slot>_bag.pkl. Our own trained models, ~2 MB each, so they ship IN the
# repository; their sha256 is in WEIGHTS_MANIFEST and is checked before any bag is unpickled.
SELECTOR_BAGS = Path(os.environ.get("FILAMENT_SELECTOR_BAGS", REPO / "weights" / "selector"))
WEIGHTS_MANIFEST = REPO / "weights" / "MANIFEST.sha256"
REFERENCE_SUBMISSIONS = REPO / "submissions"         # the uploaded Ours-Fast submission file (never overwritten)
PREDICTIONS = Path(os.environ.get("FILAMENT_PREDICTIONS", WORK / "predictions"))
SUBMISSIONS_OUT = Path(os.environ.get("FILAMENT_SUBMISSIONS_OUT", REPO / "submissions" / "out"))
CONFIGS = REPO / "configs"

DATASET_RAW = {
    5: NNUNET_RAW / "Dataset005_FilamentFull",
    7: NNUNET_RAW / "Dataset007_FilamentVessel",
    12: NNUNET_RAW / "Dataset012_FilamentRaw707",
    36: NNUNET_RAW / "Dataset036_LimbDarkBarbClass",
}


def require(path: Path, what: str, how: str = "") -> Path:
    """Fail with an actionable message instead of a bare FileNotFoundError."""
    if not Path(path).exists():
        msg = f"{what} not found at: {path}"
        if how:
            msg += f"\n  {how}"
        msg += "\n  Override the location with the matching environment variable (see filament_paths.py)."
        raise FileNotFoundError(msg)
    return Path(path)


def magfilo_json() -> Path:
    """Kept for API compatibility with the frozen modules. Raises: no stage may read the MAGFiLO release."""
    raise RuntimeError("MAGFiLO 1.0 (contains test annotations) is never read by this repository")
