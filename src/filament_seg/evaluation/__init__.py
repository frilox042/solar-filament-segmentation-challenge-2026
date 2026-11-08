"""Offline evaluation of Ours-Fast (and of its base network D007) on the 707 out-of-fold training observations
(the paper's Evaluation section).

Not part of the CPU stage: nothing here writes a submission, and the md5 gate's code fingerprint excludes this
package. The reference is the competition's training COCO JSON, the only label file the competition ships; the
historical per-record PNG tree (`extracted/`, a PNG export) is read only under `--records png`, to
replicate reported values. The MAGFiLO release is never read.

    oof_selection       cross-fitted selector keeps and scores on the OOF rows, and the kept OOF masks
    host_rows           per-record host-notebook row counts (TP pairs, FP, FN, m2n degrees, overlapping-pair IoU)
    bootstrap           image-resampling estimators (pooled ratio of sums; mean paired difference)
    heldout_annotator   a held-out annotator record vs the models, against the remaining records
    calibration         ECE of the selector score, raw and cross-fitted isotonic
    morphometry         width, branching, split degree and barb coverage of the kept masks
    records             the per-record source switch: `coco` (default) or `png` (historical replication only)
    records_coco        per-annotator-record masks rasterised from the competition COCO (pycocotools)
    records_png         the historical per-record PNG tree (`extracted/training`, a PNG export of the COCO)
    annotator_agreement annotators' existence vs extent agreement (no model involved)
    never_seen          features of the annotated filaments a model never fires on
"""
