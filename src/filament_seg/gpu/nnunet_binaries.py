"""Run inside the nnU-Net environment: argmax / probability predictions of one nnU-Net member on a folder of cases.

The same nnUNetPredictor call `nnUNetv2_predict` makes (the recipe of every binary test member):
tile step 0.5 (nnU-Net's default), Gaussian weighting, mirror TTA, checkpoint_final, the given folds; outputs
<case>.png (argmax label map; consumers read label > 0) and, with --save-probabilities, <case>.npz.

    python -m filament_seg.gpu.nnunet_binaries --model-dir DIR --folds 0 1 2 3 4 --in IN --out OUT
           [--step 0.5] [--save-probabilities] [--device cuda|cpu] [--construct-only]
`--construct-only` builds the predictor and loads the checkpoints, then exits (the CPU dry-run test).
Resume-safe: nnU-Net's own `overwrite=False` skips cases whose outputs exist.
"""
from __future__ import annotations

import argparse
import sys


def build_predictor(model_dir: str, folds, step: float, device: str):
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    p = nnUNetPredictor(tile_step_size=step, use_gaussian=True, use_mirroring=True,
                        perform_everything_on_device=True, device=torch.device(device),
                        verbose=False, verbose_preprocessing=False, allow_tqdm=False)
    p.initialize_from_trained_model_folder(model_dir, tuple(folds), checkpoint_name="checkpoint_final.pth")
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--folds", type=int, nargs="+", required=True)
    ap.add_argument("--in", dest="inp")
    ap.add_argument("--out")
    ap.add_argument("--step", type=float, default=0.5)
    ap.add_argument("--save-probabilities", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--npp", type=int, default=3)
    ap.add_argument("--nps", type=int, default=3)
    ap.add_argument("--construct-only", action="store_true")
    a = ap.parse_args(argv)
    p = build_predictor(a.model_dir, a.folds, a.step, a.device)
    if a.construct_only:
        print(f"OK predictor: {len(p.list_of_parameters)} fold(s), trainer {p.trainer_name}, "
              f"patch {p.configuration_manager.patch_size}, mirroring axes {p.allowed_mirroring_axes}")
        return 0
    p.predict_from_files(a.inp, a.out, save_probabilities=a.save_probabilities, overwrite=False,
                         num_processes_preprocessing=a.npp, num_processes_segmentation_export=a.nps,
                         folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
