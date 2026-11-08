"""The submission writer (`filament_seg.submission.rle`) and gate (`...verify`).

Round trip: random non-touching blobs per image -> PNG -> CSV -> decode every row with pycocotools -> the decoded
instances equal the 4-connected components >= min_area of the PNG, rows of one image never overlap, LF line endings,
no quotes, and the verify gate passes. Negative controls: a CSV with a duplicated (overlapping) row and a CSV with a
CR line ending must both FAIL the verify gate.
"""
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filament_seg.submission import rle, verify  # noqa: E402


def _decode(counts):
    from pycocotools import mask as M
    return M.decode({"size": [2048, 2048], "counts": counts.encode()}).astype(bool)


def _images(tmp_path, rng):
    d = tmp_path / "by_image"
    d.mkdir()
    truth = {}
    for name in ("20110101000000Ch", "20110102000000Mh"):
        m = np.zeros((2048, 2048), bool)
        for _ in range(6):
            y, x = rng.integers(50, 1950, size=2)
            m[y:y + int(rng.integers(3, 40)), x:x + int(rng.integers(3, 40))] = True
        Image.fromarray(m.astype(np.uint8) * 255).save(d / f"{name}.png")
        lab, n = ndimage.label(m)
        truth[name] = [lab == i for i in range(1, n + 1) if (lab == i).sum() >= 20]
    Image.fromarray(np.zeros((2048, 2048), np.uint8)).save(d / "20110103000000Th.png")   # an empty image
    return d, truth


def _rows(path):
    with open(path, newline="") as f:
        r = list(csv.reader(f))
    assert r[0] == ["filament_id", "segmentation_rle"]
    return r[1:]


def _verify(csv_path, png_dir):
    try:
        return int(verify.main([str(csv_path), str(png_dir), "--min-area", "20"]) or 0)
    except SystemExit as e:
        return int(e.code or 0)


def test_round_trip_and_no_overlap(tmp_path):
    d, truth = _images(tmp_path, np.random.default_rng(1))
    out = tmp_path / "sub.csv"
    rle.write_csv(d, out, min_area=20)
    raw = out.read_bytes()
    assert b"\r" not in raw and b'"' not in raw
    rows = _rows(out)
    by_img = {}
    for fid, counts in rows:
        img, n = fid.rsplit("_", 1)
        by_img.setdefault(img, []).append(_decode(counts))
    assert set(by_img) == set(k for k, v in truth.items() if v)        # the empty image has no row
    for img, masks in by_img.items():
        assert len(masks) == len(truth[img])
        for a, b in zip(masks, truth[img], strict=True):
            assert np.array_equal(a, b)
        tot = np.sum(masks, axis=0)
        assert tot.max() == 1                                            # rows of one image never overlap
    assert _verify(out, d) == 0


def test_negative_controls_fail_the_gate(tmp_path):
    d, _ = _images(tmp_path, np.random.default_rng(2))
    good = tmp_path / "good.csv"
    rle.write_csv(d, good, min_area=20)
    rows = _rows(good)
    dup = tmp_path / "dup.csv"
    with open(dup, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["filament_id", "segmentation_rle"])
        w.writerows(rows + [(rows[0][0].rsplit("_", 1)[0] + "_99", rows[0][1])])      # overlaps row 0
    assert _verify(dup, d) != 0
    crlf = tmp_path / "crlf.csv"
    crlf.write_bytes(good.read_bytes().replace(b"\n", b"\r\n"))
    assert _verify(crlf, d) != 0
