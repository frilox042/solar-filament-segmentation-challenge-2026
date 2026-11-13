# Licence of the released weights

The code in this repository is MIT (`LICENSE`). The released weights are MIT as well, subject to the notice below.

| files (layout paths, as in `MANIFEST.sha256`) | licence |
|---|---|
| `nnUNet_results/**`: the two nnU-Net networks of nnHALO (D007, D036; fold 0), with their `plans.json`, `dataset.json` and `dataset_fingerprint.json` | **MIT** (`LICENSE`), subject to the notice below |
| `selector/nnHALO_bag.pkl`: the per-object selector bag | **MIT** (`LICENSE`), subject to the notice below |

## Notice about the training data

These files are our own trained models, trained from scratch (no pretrained initialisation). They are released under
the MIT licence in `LICENSE`.

They were trained on the Competition Data of the Solar Filament Segmentation Challenge 2026. The competition rules
(§4a) say:

> You may access and use the Competition Data for non-commercial purposes only, including for participating in the
> Competition … and for academic research and education.

The MIT licence covers our copyright in the weights. It does not grant any right in the Competition Data, and it does
not settle whether a given use of models trained on that data is compatible with the competition rules. **Users are
responsible for making sure their own use is permitted.**

## The competition's winner-licence condition

The competition rules ask winners to release their submission under an OSI-approved licence that does not limit
commercial use. The code, the nnU-Net weights and the selector bag are all MIT, which satisfies the condition.
