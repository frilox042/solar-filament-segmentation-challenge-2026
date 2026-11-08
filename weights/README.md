# Weights: inventory

**Stored in this repository: Ours-Fast.** Its two networks (the nnU-Nets D007 and D036, fold 0; 0.474 GB, 8 files
with their metadata) are in `fast/`. Each checkpoint is stored as parts of at most
90 MiB (`checkpoint_final.pth.part000`, ...) because GitHub blocks files above 100 MiB.
`python scripts/assemble_weights.py` joins them into `$FILAMENT_WEIGHTS` and checks every file against
`MANIFEST.sha256`; `scripts/predict.py --from-images` and `pipeline.ipynb` do the same automatically. The selector bag
is `selector/FAST_bag.pkl`.

**Licence:** `LICENSE-WEIGHTS.md`. The nnU-Net weights and the selector bag are MIT (trained on the non-commercial
Competition Data; see the notice there).

All checkpoints are `checkpoint_final.pth`. Source hashes (sha256 of the training-run files) are in
`SOURCE_CHECKPOINTS.sha256`, with paths relative to the training working directory.

| network | architecture / trainer | fold | size per checkpoint (stored / shipped) |
|---|---|---|---|
| D007 `Dataset007_FilamentVessel` (5 ch) | nnU-Net 2d, `nnUNetTrainer_250epochs` (stock nnunetv2 2.8.1), `nnUNetPlans` | 0 | 473.7 MB / 236.8 MB |
| D036 `Dataset036_LimbDarkBarbClass` (6 ch, 3 classes) | same | 0 | 473.7 MB / 236.8 MB |

Plus the nnU-Net metadata per network (`plans.json`, `dataset.json`, `dataset_fingerprint.json`; ~100 kB). The
selector's second member mask (P, the base map at p ≥ 0.5) has no weights: it is computed from the D007 output.

## Stripped nnU-Net checkpoints (what the manifest pins)

The two nnU-Net `checkpoint_final.pth` files are **inference-only copies**. Compared with the files as stored
(hashes in `SOURCE_CHECKPOINTS.sha256`), each has four top-level keys removed:

| removed key | what it is | size |
|---|---|---|
| `optimizer_state` | SGD momentum buffers, one per parameter | ~236.8 MB, half the file |
| `grad_scaler_state` | mixed-precision loss-scaler state | small |
| `logging` | the per-epoch training curves (`nnUNetLogger`) | small |
| `_best_ema` | the running best pseudo-Dice used to pick `checkpoint_best` | a float |

**Why they can go:** inference does not read them. `nnUNetPredictor.initialize_from_trained_model_folder` (nnunetv2
2.8.1) reads only `trainer_name`, `init_args['configuration']`, `inference_allowed_mirroring_axes` and
`network_weights`. The kept keys are exactly `network_weights`, `init_args`, `trainer_name`,
`inference_allowed_mirroring_axes` and `current_epoch`.

**Verified** (on every stripped checkpoint when the release was built, these two included):
- All 448 tensors of `network_weights` are `torch.equal` to the originals, with the same key set.
- Every kept key equals the original's, and the removed keys are exactly the four above.
- A CPU forward pass through `nnUNetPredictor` gave identical output from the stripped and the original files.
- All JSON metadata is **unchanged** (byte-identical copies, same hashes).
- An end-to-end GPU run of Ours-Fast from the stripped files (assembled from `fast/`) agrees with the submitted file
  on 178 of 180 images (equal counts, image Dice ≥ 0.9997); on the other two, one selector decision near the cut
  flipped (README, "Reproduction tolerance").

The CPU smoke test is no evidence about the stripped checkpoints: it starts from cached network outputs and never
loads a `.pth`.

**Consequence for training:** an interrupted training run **cannot be resumed** from these files.
`nnUNetTrainer.load_checkpoint` (`--c`) reads `logging`, `_best_ema`, `optimizer_state` and `grad_scaler_state`, and
fails without them. They can still serve as an initialisation (`-pretrained_weights`), because
`load_pretrained_weights` reads only `network_weights`. That is a reading of the nnunetv2 2.8.1 source, not a run. Full
retraining from scratch (`docs/TRAINING.md`) does not need them at all.

## Layout and integrity

`MANIFEST.sha256` lists every shipped file with its sha256 (its nnU-Net checkpoint lines are the stripped files').
It is generated mechanically from `SOURCE_CHECKPOINTS.sha256`, with the same hashes except for the two stripped
nnU-Net checkpoints; `tests/test_assemble_weights.py` checks that:

| training path (`SOURCE_CHECKPOINTS.sha256`) | layout path (`MANIFEST.sha256`) |
|---|---|
| `nnunet_env/nnUNet_results/<rest>` | `nnUNet_results/<rest>` |

`assemble_weights.py` writes this layout, which is what `filament_seg.gpu.pipeline` reads (it sets
`nnUNet_results=$FILAMENT_WEIGHTS/nnUNet_results`):

```
$FILAMENT_WEIGHTS/
  nnUNet_results/<Dataset>/nnUNetTrainer_250epochs__nnUNetPlans__2d/{plans.json,dataset.json,dataset_fingerprint.json}
  nnUNet_results/<Dataset>/nnUNetTrainer_250epochs__nnUNetPlans__2d/fold_0/checkpoint_final.pth
```

`python scripts/check_weights_tree.py <tree>` checks a tree: every manifest entry present with its sha256, nothing
unlisted (`strip_report.json` is ignored).
`tests/test_weights_tree.py` holds its negative controls and runs it on the assembled weights.

**Selector bag.** `selector/FAST_bag.pkl` (about 2 MB) is our own trained model. It is listed in the manifest, and
`filament_seg.selector_stage` checks its sha256 before unpickling it.
