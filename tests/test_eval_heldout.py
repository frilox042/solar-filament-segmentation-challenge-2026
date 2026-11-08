"""Host-row scorer, pixel Dice and bootstrap estimators of the held-out-annotator evaluation, on hand toys.

Each test has a negative control. The end-to-end gate is `scripts/eval_heldout_annotator.py`.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament_seg.evaluation import bootstrap as B  # noqa: E402
from filament_seg.evaluation import host_rows as HR  # noqa: E402


def test_row_counts_strict_iou_and_m2n():
    # one GT of area 10; prediction 0 covers 5 of it with area 5 (IoU 0.5 exactly: NOT a hit), prediction 1 disjoint
    c, iou = HR.row_counts(np.array([[5, 0]]), [10], [5, 4])
    f = dict(zip(HR.FIELDS, c, strict=True))
    assert f["tp"] == 0 and f["fp"] == 2 and f["fn"] == 1
    assert f["p0"] == 1 and f["gt1"] == 1 and f["n_pairs"] == 1 and iou.tolist() == [0.5]
    # negative control: one more pixel of overlap makes it a hit
    c2, _ = HR.row_counts(np.array([[6, 0]]), [10], [6, 4])
    assert dict(zip(HR.FIELDS, c2, strict=True))["tp"] == 1


def test_row_counts_empty_branches():
    c, _ = HR.row_counts(np.zeros((0, 3), np.int64), [], [1, 2, 3])
    assert c[HR.FI["fp"]] == 3 and c[HR.FI["fn"]] == 0
    c, _ = HR.row_counts(np.zeros((2, 0), np.int64), [1, 2], [])
    assert c[HR.FI["fn"]] == 2 and c[HR.FI["fp"]] == 0


def test_pq_from_counts():
    S = np.zeros(len(HR.FIELDS))
    S[HR.FI["s_iou_tp"]], S[HR.FI["tp"]], S[HR.FI["fp"]], S[HR.FI["fn"]] = 1.6, 2, 1, 3
    st = HR.stats(S)
    assert st["pq_w05"] == pytest.approx(1.6 / (2 + 0.5 + 1.5)) and st["pq_w10"] == pytest.approx(1.6 / 6)


def test_inter_lists_matches_label_path():
    rng = np.random.default_rng(0)
    m = np.zeros((HR.H, HR.W), bool)
    m[100:120, 200:260] = True
    m[500:503, 10:400] = True
    lab, n, _ = HR.components(m)
    P = [np.flatnonzero((lab == j).ravel()) for j in range(1, n + 1)]
    G = [np.sort(rng.choice(HR.H * HR.W, 50, replace=False)), np.flatnonzero(m.ravel())[:700]]
    assert np.array_equal(HR.inter_lists(P, G), HR.inter_label(lab, n, G))


def test_dice_fg_matches_torchmetrics():
    pytest.importorskip("torchmetrics")
    import torch
    from torchmetrics.segmentation import DiceScore
    rng = np.random.default_rng(1)
    for _ in range(3):
        p, t = rng.random((1, 64, 64)) < 0.3, rng.random((1, 64, 64)) < 0.2
        d = DiceScore(num_classes=2, include_background=False, average="micro", input_format="index")
        d.update(torch.from_numpy(p.astype(np.int64)), torch.from_numpy(t.astype(np.int64)))
        mine = HR.dice_fg((p & t).sum(), p.sum(), t.sum())
        assert abs(float(d.compute()) - float(mine)) < 1e-6
    # negative control: include_background=True is a different number
    d = DiceScore(num_classes=2, include_background=True, average="micro", input_format="index")
    d.update(torch.from_numpy(p.astype(np.int64)), torch.from_numpy(t.astype(np.int64)))
    assert abs(float(d.compute()) - float(mine)) > 0.1


def test_paired_mean_and_pooled_share_draws():
    a, b = np.arange(10.0), np.arange(10.0) - 1
    r = B.paired_mean(a, b, seed=0, nb=200)
    assert r["mean_diff"] == 1 and r["ci95"] == [1.0, 1.0]
    # pooled: identical arms give a zero delta and no preference; a better arm a positive interval
    c = np.zeros((10, len(HR.FIELDS)))
    c[:, HR.FI["tp"]], c[:, HR.FI["s_iou_tp"]], c[:, HR.FI["fn"]] = 2, 1.4, 1
    better = c.copy()
    better[:, HR.FI["fn"]] = 0
    W = B.weights(10, 0, 300)
    same = B.pooled(W, [c], [c], {"pq_w05": True})["pq_w05"]
    up = B.pooled(W, [better], [c], {"pq_w05": True})["pq_w05"]
    assert same["delta"] == 0 and not same["model_favoured"] and up["model_favoured"]
