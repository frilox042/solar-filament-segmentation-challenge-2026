#!/usr/bin/env python3
"""Write the model's OUT-OF-FOLD foreground probability for every training image, as a PNG.

The out-of-fold property is the whole point: each image's probability must come from the fold that did NOT train
on it, or the maps would leak the answer into every downstream step.

That property is guaranteed here by CONSTRUCTION rather than checked afterwards: the loop is over folds, and each
fold predicts only the stems in its own `val` list, read from `splits_final.json`.

Probabilities are written as uint16 PNGs (0..65535) rather than .npz: 707 float32 maps at 2048x2048
would be ~24 GB, and a 16-bit quantisation of [0, 1] is far finer than anything downstream can use.

    python predict_oof_probs.py --out predictions/oof_probs
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

from filament_paths import NNUNET_PREPROCESSED, NNUNET_RAW, NNUNET_RESULTS

DS = "Dataset007_FilamentVessel"
TRAINER_DIR = "nnUNetTrainer_250epochs__nnUNetPlans__2d"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--step-size", type=float, default=0.75,
                    help="Stage A measured 0.75 as flat against the deployed 0.5 at 3.1x the "
                         "speed, so it is the sensible default for a full 707-image pass.")
    ap.add_argument("--trainer-dir", default=None,
                    help="results sub-directory of the model (default: the deployed D007 trainer)")
    ap.add_argument("--model-dataset", default=DS,
                    help="nnU-Net results dataset holding the model (default Dataset007). Inputs and splits are ALWAYS "
                         "Dataset007's; only use a model whose folds hold out D007's fold images on D007's channels")
    ap.add_argument("--input-dataset", default=None,
                    help="nnU-Net RAW dataset whose imagesTr/channel_names feed the model (default: Dataset007, the "
                         "historical behaviour). Use the model's own dataset when its channel stack differs from D007's "
                         "(D005 4-ch, D012 1-ch, D036 6-ch). Splits are ALWAYS Dataset007's.")
    ap.add_argument("--fg", choices=["class1", "not_bg", "class2"], default="class1",
                    help="class1: write p[1] (default, binary models); not_bg: write 1 - p[0], i.e. every foreground "
                         "class merged (D036's filament+barb, matching its validation_binary 1|2 merge); "
                         "class2: write p[2] (D036's barb class)")
    ap.add_argument("--out-notbg", default=None,
                    help="ALSO write 1 - p[0] from the SAME forward pass into this directory (the spine mask and "
                         "the barb map must come from one pass)")
    ap.add_argument("--no-mirror-tta", action="store_true",
                    help="disable nnU-Net's mirror test-time augmentation")
    ap.add_argument("--mirror-input", action="store_true",
                    help="feed the LEFT-RIGHT mirrored image, then un-flip the output back to the original frame")
    ap.add_argument("--stems-file", default=None,
                    help="predict only the stems (case ids, one per line) listed here, within each fold's val list")
    ap.add_argument("--resume", action="store_true",
                    help="skip stems whose output PNG(s) already exist (writes are tmp+rename, so an existing file is whole)")
    ap.add_argument("--folds", type=int, nargs="*", default=None,
                    help="predict only these folds' held-out images (default: all folds)")
    a = ap.parse_args()
    trainer_dir = a.trainer_dir or TRAINER_DIR

    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    splits = json.loads((NNUNET_PREPROCESSED / DS / "splits_final.json").read_text())
    in_ds = a.input_dataset or DS
    raw_dir = NNUNET_RAW / in_ds / "imagesTr"
    n_ch = len(json.loads(((NNUNET_PREPROCESSED / DS / "dataset.json") if in_ds == DS
                           else (NNUNET_RAW / in_ds / "dataset.json")).read_text())["channel_names"])
    os.makedirs(a.out, exist_ok=True)
    if a.out_notbg:
        os.makedirs(a.out_notbg, exist_ok=True)
    only = set(open(a.stems_file).read().split()) if a.stems_file else None

    written = skipped = 0
    # default: every fold that has a trained model (D007's splits file has more entries than 5 trained folds)
    folds = a.folds if a.folds is not None else [f for f in range(len(splits)) if (NNUNET_RESULTS / a.model_dataset / trainer_dir
                                                                                  / f"fold_{f}" / "checkpoint_final.pth").exists()]
    for fold in folds:
        pred = nnUNetPredictor(tile_step_size=a.step_size, use_gaussian=True, use_mirroring=not a.no_mirror_tta,
                               device=torch.device("cuda"), verbose=False,
                               verbose_preprocessing=False, allow_tqdm=False)
        pred.initialize_from_trained_model_folder(str(NNUNET_RESULTS / a.model_dataset / trainer_dir), (fold,),
                                                  checkpoint_name="checkpoint_final.pth")
        for stem in splits[fold]["val"]:          # ONLY this fold's held-out images
            if only is not None and stem not in only:
                continue
            outs = [os.path.join(a.out, f"{stem}.png")] + ([os.path.join(a.out_notbg, f"{stem}.png")] if a.out_notbg else [])
            if a.resume and all(os.path.exists(o) for o in outs):
                skipped += 1
                continue
            chans = []
            for c in range(n_ch):
                arr = np.asarray(Image.open(raw_dir / f"{stem}_{c:04d}.png"))
                chans.append((arr[..., 0] if arr.ndim == 3 else arr).astype(np.float32))
            x = np.stack(chans)[:, None]
            if a.mirror_input:
                x = np.ascontiguousarray(x[..., ::-1])
            _, probs = pred.predict_single_npy_array(x, {"spacing": [999.0, 1.0, 1.0]},
                                                     None, None, True)
            p = np.asarray(probs)
            cls = lambda k: p[k, 0] if p.ndim == 4 else p[k]
            if a.fg == "class1":
                fg = cls(1)     # class 1 = filament
            elif a.fg == "class2":
                fg = cls(2)     # class 2 = barb (D036)
            else:
                fg = 1.0 - cls(0)   # every non-background class
            maps = [fg] + ([1.0 - cls(0)] if a.out_notbg else [])
            for o, mp in zip(outs, maps):
                if a.mirror_input:
                    mp = mp[..., ::-1]               # back to the ORIGINAL frame
                tmp = o + ".part"                    # not *.png: a crash never leaves a globbable partial
                Image.fromarray((np.clip(mp, 0, 1) * 65535).astype(np.uint16)).save(tmp, format="PNG")
                os.replace(tmp, o)
            written += 1
            if written % 50 == 0:
                print(f"  {written} images", flush=True)
        del pred
        torch.cuda.empty_cache()

    total = sum(len([s for s in splits[f]["val"] if only is None or s in only]) for f in folds)
    if written + skipped != total:
        raise SystemExit(f"wrote {written} probability maps but the splits hold {total} images; "
                         f"the cascade dataset would be built on a partial set.")
    print(f"wrote {written} out-of-fold probability maps to {a.out} (skipped {skipped} existing)")


if __name__ == "__main__":
    main()
