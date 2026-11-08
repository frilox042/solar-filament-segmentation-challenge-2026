"""Input channels of the nnU-Net datasets, from one raw 8-bit H-alpha frame.

The channel builders as pure functions; `prep/labels.py` builds the labels. Each channel is a uint16 image; nnU-Net z-scores every
channel per image.

  D005 (4 ch): raw, CLAHE (kernel 256, clip 0.01), raw - Gaussian(sigma 64), structure-tensor coherence (sigma 3)
  D012 (1 ch): D005 channel 0
  D007 (5 ch): D005's four + single-scale Sato dark ridge (sigma 16; off-disk zeroed, disk = raw > 0.08 max)
  D036 (6 ch): D007's five + limb-darkening-corrected raw (quadratic law 1 - 0.077 (1 - mu) - 0.358 (1 - mu)^2,
               R = 903.2 px, centre = filled largest component of raw > 40)
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter

BGSUB_SIGMA, COH_SIGMA, SATO_SIGMA = 64.0, 3.0, 16.0
# Quadratic H-alpha limb-darkening law (historical constants, measured early in the
# project on raw GONG frames). The textbook continuum law would over-correct the
# H-alpha limb about twofold. Kept fixed: refitting moves the curve by < 1% inside
# r/R 0.97, which did not justify retraining the model that uses this channel.
A1, A2 = 0.077, 0.358
R_LIMB = 903.2                 # limb-fit radius in px


def to_u16(x: np.ndarray) -> np.ndarray:
    """Per-image min-max to uint16 (D005 / D007 convention)."""
    x = x.astype(np.float32)
    lo, hi = float(x.min()), float(x.max())
    return ((x - lo) / (hi - lo + 1e-8) * 65535.0).astype(np.uint16)


def to_u16_nan(x: np.ndarray) -> np.ndarray:
    """Per-image min-max to uint16, NaN-safe (the limb-dark channel's convention)."""
    x = np.asarray(x, np.float32)
    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    if not np.isfinite(lo) or hi <= lo:
        return np.zeros(x.shape, np.uint16)
    return ((x - lo) / (hi - lo) * 65535.0).astype(np.uint16)


def four_channels(raw_u8: np.ndarray, sigma: float = BGSUB_SIGMA, coh_sigma: float = COH_SIGMA) -> tuple:
    """D005's channels 0-3 from the raw 8-bit frame."""
    from skimage.exposure import equalize_adapthist
    from skimage.feature import structure_tensor, structure_tensor_eigenvalues
    raw = raw_u8.astype(np.float32)
    ch0 = to_u16(raw)
    ch1 = to_u16(equalize_adapthist(raw_u8, kernel_size=256, clip_limit=0.01))
    ch2 = to_u16(raw - gaussian_filter(raw, sigma=sigma))
    Axx, Axy, Ayy = structure_tensor(raw, sigma=coh_sigma, order="rc")
    l1, l2 = structure_tensor_eigenvalues((Axx, Axy, Ayy))
    coh = (l1 - l2) / (l1 + l2 + 1e-6)
    ch3 = to_u16(coh)
    return ch0, ch1, ch2, ch3


def sato_channel(ch0_u16: np.ndarray, sigma: float = SATO_SIGMA) -> np.ndarray:
    """D007's channel 4: Sato dark-ridge response at one scale, zero off the (threshold) disk."""
    from skimage.filters import sato
    raw = ch0_u16.astype(np.float32)
    disk = raw > 0.08 * float(raw.max())
    resp = sato(raw, sigmas=[sigma], black_ridges=True, mode="reflect").astype(np.float32)
    resp[~disk] = 0.0
    return to_u16(resp)


def fit_disk(img: np.ndarray) -> tuple[float, float]:
    """Disk centre: centroid of the filled largest component of img > 40."""
    m = img > 40
    lab, n = ndimage.label(m)
    if n == 0:
        return img.shape[0] / 2, img.shape[1] / 2
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    m = ndimage.binary_fill_holes(lab == (int(np.argmax(sizes)) + 1))
    ys, xs = np.where(m)
    return float(ys.mean()), float(xs.mean())


def limb_field(shape: tuple, cy: float, cx: float, R: float = R_LIMB) -> np.ndarray:
    """The limb-darkening law on the image grid (held at the limb value off-disk)."""
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
    rr = np.hypot(yy - cy, xx - cx) / R
    mu = np.sqrt(np.clip(1.0 - rr ** 2, 0.0, 1.0))
    f = 1.0 - A1 * (1.0 - mu) - A2 * (1.0 - mu) ** 2
    f[rr >= 1.0] = 1.0 - A1 - A2
    return f.astype(np.float32)


def limbdark_channel(ch0_u16: np.ndarray) -> np.ndarray:
    """D036's channel 5 from D007's channel 0 (uint16)."""
    raw = ch0_u16.astype(np.float32)
    cy, cx = fit_disk(raw)
    return to_u16_nan(raw / np.maximum(limb_field(raw.shape, cy, cx), 1e-3))


def all_channels(raw_u8: np.ndarray) -> dict:
    """{dataset: [uint16 channel arrays]} for D005, D007, D012, D036 from one raw 8-bit frame."""
    c4 = list(four_channels(raw_u8))
    s = sato_channel(c4[0])
    ld = limbdark_channel(c4[0])
    return {"Dataset005_FilamentFull": c4, "Dataset012_FilamentRaw707": [c4[0]],
            "Dataset007_FilamentVessel": c4 + [s], "Dataset036_LimbDarkBarbClass": c4 + [s, ld]}
