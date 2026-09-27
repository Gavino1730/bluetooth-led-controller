from __future__ import annotations

"""Perceptual screen → ambient scene estimate for a single-color wall strip.

Pipeline: frame → letterbox crop → linear light → edge-weighted scene stats
→ chromaticity + luminance (separate) → highlight blend → SampleResult
"""

from dataclasses import dataclass

import numpy as np

from rgb_lights.color_matrix import linear_to_srgb


@dataclass
class SampleResult:
    """Ambient target derived from the frame (not a plain RGB average)."""

    rgb: tuple[float, float, float]  # chromaticity as sRGB 0–255 (full value)
    luma: float  # scene luminance 0–1 (linear-ish)
    saturation: float
    crop: tuple[int, int, int, int]
    highlight_luma: float = 0.0
    black: bool = False


class ScreenSampler:
    """Estimate what ambient light a movie frame should cast into the room."""

    def __init__(
        self,
        edge_depth: float = 0.32,
        blackbar: bool = True,
        blackbar_mode: str = "letterbox",
        blackbar_threshold: float = 0.07,
        blackbar_hold: int = 6,
        highlight_mix: float = 0.25,
        luma_gamma: float = 0.65,
        black_luma: float = 0.028,
    ) -> None:
        # edge_depth = how much of the frame is treated as "off-screen bleed"
        self.edge_depth = float(np.clip(edge_depth, 0.15, 0.45))
        self.blackbar = blackbar
        self.blackbar_mode = (blackbar_mode or "letterbox").lower().strip()
        self.blackbar_threshold = float(np.clip(blackbar_threshold, 0.02, 0.25))
        self.blackbar_hold = max(1, int(blackbar_hold))
        self.highlight_mix = float(np.clip(highlight_mix, 0.0, 0.45))
        self.luma_gamma = float(np.clip(luma_gamma, 0.35, 1.2))
        self.black_luma = float(np.clip(black_luma, 0.01, 0.08))
        self._stable_crop = (0, 0, 0, 0)
        self._pending_crop = (0, 0, 0, 0)
        self._pending_count = 0

    def sample(self, frame: np.ndarray) -> SampleResult:
        """frame: HxWx3 RGB uint8 (already downscaled)."""
        h, w = frame.shape[:2]
        crop = self._detect_blackbars(frame) if self.blackbar else (0, 0, 0, 0)
        top, bottom, left, right = crop
        y1, y2 = top, h - bottom
        x1, x2 = left, w - right
        if y2 <= y1 + 2 or x2 <= x1 + 2:
            content = frame
            crop = (0, 0, 0, 0)
        else:
            content = frame[y1:y2, x1:x2]

        ch, cw = content.shape[:2]
        # Stride if still large — keep ~64×36 class resolution
        step = max(1, min(ch, cw) // 48)
        grid = content[::step, ::step].astype(np.float32) / 255.0
        gh, gw = grid.shape[:2]
        if gh < 2 or gw < 2:
            return SampleResult(rgb=(0.0, 0.0, 0.0), luma=0.0, saturation=0.0, crop=crop, black=True)

        # Linear light
        lin = _srgb_to_linear_arr(grid)
        luma = 0.2126 * lin[:, :, 0] + 0.7152 * lin[:, :, 1] + 0.0722 * lin[:, :, 2]

        # Spatial weights: favor outer ring (room bleed), soft center
        yy = np.linspace(-1.0, 1.0, gh, dtype=np.float32)[:, None]
        xx = np.linspace(-1.0, 1.0, gw, dtype=np.float32)[None, :]
        # 0 at center, 1 at edges
        radial = np.maximum(np.abs(xx), np.abs(yy))
        # Outer ~edge_depth band gets full weight; center downweighted
        edge_start = 1.0 - 2.0 * self.edge_depth  # ~0.36 for depth 0.32
        spatial = np.clip((radial - edge_start) / max(1e-6, 1.0 - edge_start), 0.0, 1.0)
        spatial = 0.18 + 0.82 * (spatial ** 1.15)

        # Ignore near-black pixels (letterbox leftovers / crushed shadows)
        alive = luma > (self.black_luma * 0.35)
        if not np.any(alive):
            return SampleResult(
                rgb=(0.0, 0.0, 0.0),
                luma=0.0,
                saturation=0.0,
                crop=crop,
                highlight_luma=0.0,
                black=True,
            )

        # Scene luminance (what the room should feel)
        w_scene = spatial * alive.astype(np.float32)
        scene_luma = float((luma * w_scene).sum() / max(float(w_scene.sum()), 1e-8))

        # Chromaticity weighted by L^gamma (bright objects influence more)
        chroma = lin / np.maximum(luma[:, :, None], 1e-6)
        w_color = w_scene * np.power(np.maximum(luma, 1e-6), self.luma_gamma)
        # Mild chroma preference so gray walls don't dominate
        sat = _linear_sat(lin)
        w_color = w_color * (0.40 + 0.60 * sat)

        scene_chroma = _weighted_chroma(chroma, w_color)

        # Highlights: top luminous percentile (explosions, sun, headlights)
        highlight_luma = 0.0
        hi_chroma = scene_chroma
        if self.highlight_mix > 0.01 and np.any(alive):
            thr = float(np.percentile(luma[alive], 92))
            hi_mask = alive & (luma >= max(thr, scene_luma * 1.8, 0.12))
            if np.any(hi_mask):
                w_hi = spatial * hi_mask.astype(np.float32) * np.power(
                    np.maximum(luma, 1e-6), self.luma_gamma + 0.25
                )
                hi_chroma = _weighted_chroma(chroma, w_hi)
                highlight_luma = float(luma[hi_mask].mean())

        mix = self.highlight_mix
        # More highlight pull when highlights are much brighter than the scene
        if highlight_luma > scene_luma * 2.5:
            mix = min(0.40, mix + 0.08)
        blended = (1.0 - mix) * scene_chroma + mix * hi_chroma
        blended = blended / max(float(blended.sum()), 1e-8)

        # Encode chromaticity as sRGB at unit brightness for downstream
        rgb = _chroma_to_srgb255(blended)
        sat_out = float(np.clip(np.max(blended) - np.min(blended), 0.0, 1.0))
        is_black = scene_luma < self.black_luma

        return SampleResult(
            rgb=rgb,
            luma=scene_luma,
            saturation=sat_out,
            crop=crop,
            highlight_luma=highlight_luma,
            black=is_black,
        )

    def _detect_blackbars(self, frame: np.ndarray) -> tuple[int, int, int, int]:
        h, w = frame.shape[:2]
        # Use linear-ish luma for bar detection
        srgb = frame.astype(np.float32) / 255.0
        luma = 0.2126 * srgb[:, :, 0] + 0.7152 * srgb[:, :, 1] + 0.114 * srgb[:, :, 2]
        thr = self.blackbar_threshold

        row_mean = luma.mean(axis=1)
        top = 0
        while top < h // 3 and row_mean[top] < thr:
            top += 1
        bottom = 0
        while bottom < h // 3 and row_mean[h - 1 - bottom] < thr:
            bottom += 1

        left = right = 0
        if self.blackbar_mode == "full":
            col_mean = luma.mean(axis=0)
            while left < w // 3 and col_mean[left] < thr:
                left += 1
            while right < w // 3 and col_mean[w - 1 - right] < thr:
                right += 1

        if top + bottom >= h * 0.7:
            top = bottom = 0
        if left + right >= w * 0.7:
            left = right = 0

        detected = (top, bottom, left, right)
        if detected == self._stable_crop:
            self._pending_count = 0
            self._pending_crop = detected
            return self._stable_crop

        if detected == self._pending_crop:
            self._pending_count += 1
        else:
            self._pending_crop = detected
            self._pending_count = 1

        if self._pending_count >= self.blackbar_hold:
            self._stable_crop = detected
            self._pending_count = 0
        return self._stable_crop


def _srgb_to_linear_arr(srgb: np.ndarray) -> np.ndarray:
    """Vectorized sRGB (0–1) → linear."""
    a = srgb
    return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4).astype(np.float32)


def _linear_sat(lin: np.ndarray) -> np.ndarray:
    mx = np.maximum(np.maximum(lin[:, :, 0], lin[:, :, 1]), lin[:, :, 2])
    mn = np.minimum(np.minimum(lin[:, :, 0], lin[:, :, 1]), lin[:, :, 2])
    return np.where(mx > 1e-5, (mx - mn) / np.maximum(mx, 1e-5), 0.0).astype(np.float32)


def _weighted_chroma(chroma: np.ndarray, weights: np.ndarray) -> np.ndarray:
    w = np.maximum(weights, 0.0)
    wsum = float(w.sum())
    if wsum < 1e-8:
        return np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float32)
    c = (chroma * w[:, :, None]).sum(axis=(0, 1)) / wsum
    c = np.maximum(c, 0.0)
    s = float(c.sum())
    if s < 1e-8:
        return np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float32)
    return (c / s).astype(np.float32)


def _chroma_to_srgb255(chroma: np.ndarray) -> tuple[float, float, float]:
    """Map linear chromaticity (sums to 1) to an sRGB color at moderate value."""
    # Scale so max channel ≈ 1 in linear, then encode
    peak = float(np.max(chroma)) + 1e-8
    lin = chroma / peak
    return (
        linear_to_srgb(float(lin[0])) * 255.0,
        linear_to_srgb(float(lin[1])) * 255.0,
        linear_to_srgb(float(lin[2])) * 255.0,
    )
