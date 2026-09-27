from __future__ import annotations

"""Phone/LED color correction matrix helpers.

Fit a 3x3 matrix so commanded LED colors match what the camera measures,
then apply that matrix in movie mode for full-gamut accuracy (not just R/G/B gains).
"""

import math
from typing import Sequence

import numpy as np

IDENTITY_9: list[float] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

# Full gamut patches for auto-tune (Hyperion-style primary + secondary + white)
CAL_PATCHES: list[tuple[str, tuple[int, int, int]]] = [
    ("white", (255, 255, 255)),
    ("red", (220, 25, 25)),
    ("green", (25, 210, 55)),
    ("blue", (35, 70, 230)),
    ("cyan", (20, 200, 210)),
    ("magenta", (220, 35, 180)),
    ("yellow", (235, 205, 25)),
    ("orange", (235, 120, 25)),
]


def as_matrix9(values: Sequence[float] | None) -> list[float]:
    if not values or len(values) != 9:
        return list(IDENTITY_9)
    out = [float(v) for v in values]
    # Reject NaNs / absurd fits
    if any(not math.isfinite(v) or abs(v) > 4.0 for v in out):
        return list(IDENTITY_9)
    return out


def is_identity(matrix9: Sequence[float], eps: float = 1e-3) -> bool:
    m = as_matrix9(matrix9)
    return all(abs(a - b) <= eps for a, b in zip(m, IDENTITY_9, strict=True))


def srgb_to_linear(x: float) -> float:
    x = max(0.0, min(1.0, x))
    if x <= 0.04045:
        return x / 12.92
    return ((x + 0.055) / 1.055) ** 2.4


def linear_to_srgb(x: float) -> float:
    x = max(0.0, min(1.0, x))
    if x <= 0.0031308:
        return 12.92 * x
    return 1.055 * (x ** (1.0 / 2.4)) - 0.055


def rgb_to_linear(rgb: Sequence[float]) -> np.ndarray:
    return np.array(
        [srgb_to_linear(float(rgb[0]) / 255.0), srgb_to_linear(float(rgb[1]) / 255.0), srgb_to_linear(float(rgb[2]) / 255.0)],
        dtype=np.float64,
    )


def linear_to_rgb255(lin: np.ndarray) -> tuple[float, float, float]:
    return (
        linear_to_srgb(float(lin[0])) * 255.0,
        linear_to_srgb(float(lin[1])) * 255.0,
        linear_to_srgb(float(lin[2])) * 255.0,
    )


def luma_linear(lin: np.ndarray) -> float:
    return float(0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2])


def exposure_match(measured_lin: np.ndarray, target_lin: np.ndarray) -> np.ndarray:
    """Remove iPhone auto-exposure by matching luminance to the target patch."""
    ml = luma_linear(measured_lin)
    tl = luma_linear(target_lin)
    if ml < 1e-5:
        return measured_lin.copy()
    return measured_lin * (tl / ml)


def apply_matrix_rgb(
    r: float,
    g: float,
    b: float,
    matrix9: Sequence[float] | None,
) -> tuple[float, float, float]:
    m = as_matrix9(matrix9)
    if is_identity(m):
        return r, g, b
    mat = np.array(m, dtype=np.float64).reshape(3, 3)
    lin = rgb_to_linear((r, g, b))
    out = mat @ lin
    out = np.clip(out, 0.0, 1.5)  # allow slight headroom before sRGB encode
    rr, gg, bb = linear_to_rgb255(np.clip(out, 0.0, 1.0))
    return rr, gg, bb


def fit_color_matrix(
    targets_rgb: Sequence[Sequence[float]],
    measured_rgb: Sequence[Sequence[float]],
    *,
    ridge: float = 0.08,
) -> tuple[list[float], float]:
    """Least-squares Mat where Mat @ measured ≈ target (linear RGB).

    Returns (matrix9, mean_abs_error_0to1).
    """
    if len(targets_rgb) < 3 or len(targets_rgb) != len(measured_rgb):
        return list(IDENTITY_9), 1.0

    t_rows: list[np.ndarray] = []
    m_rows: list[np.ndarray] = []
    for t_rgb, m_rgb in zip(targets_rgb, measured_rgb, strict=True):
        t = rgb_to_linear(t_rgb)
        m = rgb_to_linear(m_rgb)
        if luma_linear(m) < 0.02:
            continue  # skip failed samples
        m = exposure_match(m, t)
        t_rows.append(t)
        m_rows.append(m)

    if len(t_rows) < 3:
        return list(IDENTITY_9), 1.0

    T = np.stack(t_rows, axis=1)  # 3 x N
    M = np.stack(m_rows, axis=1)  # 3 x N

    # Ridge toward identity: solve Mat @ M ≈ T with soft prior Mat ≈ I
    # Using: Mat = T M+ with Tikhonov on columns
    eye = np.eye(3)
    # Augment: [M | sqrt(ridge)*I] and [T | sqrt(ridge)*I]
    M_aug = np.concatenate([M, math.sqrt(ridge) * eye], axis=1)
    T_aug = np.concatenate([T, math.sqrt(ridge) * eye], axis=1)
    Mat, *_ = np.linalg.lstsq(M_aug.T, T_aug.T, rcond=None)
    Mat = Mat.T  # 3x3

    # Stability clamps
    Mat = np.clip(Mat, -0.35, 2.4)
    # Keep diagonal dominant-ish
    for i in range(3):
        if Mat[i, i] < 0.35:
            Mat[i, i] = 0.35

    pred = Mat @ M
    err = float(np.mean(np.abs(pred - T)))
    flat = [float(x) for x in Mat.reshape(-1)]
    return flat, err


def matrix_to_gains(matrix9: Sequence[float]) -> tuple[float, float, float]:
    """Approximate diagonal gains for UI display."""
    m = as_matrix9(matrix9)
    return (
        float(np_clip(m[0], 0.5, 1.5)),
        float(np_clip(m[4], 0.5, 1.5)),
        float(np_clip(m[8], 0.5, 1.5)),
    )


def np_clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))
