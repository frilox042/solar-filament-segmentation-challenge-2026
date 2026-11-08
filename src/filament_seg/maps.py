"""Reading and writing probability maps and binary masks (the conventions every stored map follows)."""
from __future__ import annotations

import numpy as np
from PIL import Image

F65535 = np.float32(65535.0)


def load_bin(path) -> np.ndarray:
    """Bool foreground mask of a binary PNG (> 0; channel 0 if the PNG is RGB)."""
    a = np.asarray(Image.open(path))
    return (a[..., 0] if a.ndim == 3 else a) > 0


def load_u16(path) -> np.ndarray:
    """Read a 2048x2048 uint16 probability PNG (asserts dtype and shape)."""
    a = np.asarray(Image.open(path))
    assert a.dtype == np.uint16 and a.shape == (2048, 2048), (path, a.dtype, a.shape)
    return a


def pr_of(u16: np.ndarray) -> np.ndarray:
    """Probability of a uint16 map: u16 / 65535 in float32 (the inverse of the map writer)."""
    return u16.astype(np.float32) / 65535.0


def load_prob_png(path) -> np.ndarray:
    """An OOF uint16 probability PNG as float32 probability (no shape assertion; = the OOF row builder's read)."""
    return np.asarray(Image.open(path)).astype(np.float32) / 65535.0


def quantise_like_oof(fg: np.ndarray) -> np.ndarray:
    """The map writer truncates: (clip(fg, 0, 1) * 65535).astype(uint16); readers divide by 65535.
    A float test map gets the identical round trip so the same threshold means the same arithmetic."""
    return (np.clip(fg, 0, 1) * 65535).astype(np.uint16).astype(np.float32) / 65535.0


def npz_foreground(path) -> np.ndarray:
    """Foreground probability (class 1) from an nnU-Net `--save_probabilities` npz ((2, 1, H, W) or (2, H, W))."""
    fg = np.load(path)["probabilities"]
    return fg[1, 0] if fg.ndim == 4 else fg[1]
