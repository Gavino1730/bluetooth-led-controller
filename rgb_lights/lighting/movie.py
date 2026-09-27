from __future__ import annotations

"""Cinema ambient engine for a single wall-wrap strip (dorm-friendly).

Separates chromaticity from luminance, maps scene brightness to comfortable
room light, and uses asymmetric temporal filtering so flashes attack fast
and darkness settles slowly — like real light, not a gaming peripheral.
"""

import colorsys
import math
import time
from dataclasses import dataclass
from typing import Sequence

from rgb_lights.color_matrix import apply_matrix_rgb, as_matrix9, is_identity, linear_to_srgb, srgb_to_linear
from rgb_lights.screen.sampler import SampleResult


# name -> (room_scale, sat_gain, black_lift, attack_s, release_s, color_s)
# room_scale multiplies calibrated target luminance (not raw RGB).
_PRESETS = {
    "subtle": (0.32, 1.12, 0.00, 0.045, 0.70, 0.22),
    "cinema": (0.48, 1.22, 0.00, 0.035, 0.55, 0.16),
    "moderate": (0.48, 1.22, 0.00, 0.035, 0.55, 0.16),  # alias → cinema
    "immersive": (0.68, 1.32, 0.01, 0.028, 0.40, 0.12),
    "high": (0.68, 1.32, 0.01, 0.028, 0.40, 0.12),  # alias
    "maximum": (0.92, 1.45, 0.02, 0.020, 0.28, 0.08),
    "extreme": (0.92, 1.45, 0.02, 0.020, 0.28, 0.08),  # alias
}


@dataclass
class MovieFrame:
    rgb: tuple[int, int, int]
    luma: float
    saturation: float
    intensity: str
    cut: bool = False


class MovieEngine:
    """Perceptual ambient mapper for wall LEDs around a dorm room."""

    def __init__(
        self,
        intensity: str = "cinema",
        saturation: float | None = None,
        min_brightness: float = 0.0,
        max_brightness: float | None = None,
        smoothing: float | None = None,
        global_intensity: float = 0.85,
        r_gain: float = 1.0,
        g_gain: float = 1.0,
        b_gain: float = 1.0,
        color_matrix: Sequence[float] | None = None,
        black_luma: float = 0.028,
    ) -> None:
        name = (intensity or "cinema").lower().strip()
        if name not in _PRESETS:
            name = "cinema"
        self.intensity_name = "cinema" if name == "moderate" else (
            "immersive" if name == "high" else ("maximum" if name == "extreme" else name)
        )
        room, sat, black_lift, attack, release, color_tau = _PRESETS[name]

        # Optional legacy smoothing override → stretch all taus
        if smoothing is not None and smoothing > 0:
            stretch = 0.5 + float(smoothing)
            attack *= stretch
            release *= stretch
            color_tau *= stretch

        self.room_scale = float(room) * float(_clamp01(global_intensity if max_brightness is None else max_brightness))
        self.sat_gain = float(saturation if saturation is not None else sat)
        self.black_lift = float(black_lift)
        self.min_brightness = float(_clamp01(min_brightness))
        self.attack_s = float(attack)
        self.release_s = float(release)
        self.color_s = float(color_tau)
        self.black_luma = float(black_luma)
        self.r_gain = float(r_gain)
        self.g_gain = float(g_gain)
        self.b_gain = float(b_gain)
        self.color_matrix = as_matrix9(color_matrix)

        # For CLI status (compatible field name)
        self.max_brightness = self.room_scale

        self._h = 0.0
        self._s = 0.0
        self._bri = 0.0
        self._primed = False
        self._last_t = time.perf_counter()
        self._prev_luma = 0.0
        self._prev_chroma = (0.33, 0.33, 0.33)

    def process(self, sample: SampleResult) -> MovieFrame:
        now = time.perf_counter()
        dt = max(1e-3, min(0.25, now - self._last_t))
        self._last_t = now

        # --- Target chromaticity ---
        r, g, b = sample.rgb
        if not is_identity(self.color_matrix):
            r, g, b = apply_matrix_rgb(r, g, b, self.color_matrix)
        elif abs(self.r_gain - 1.0) > 0.01 or abs(self.g_gain - 1.0) > 0.01 or abs(self.b_gain - 1.0) > 0.01:
            r, g, b = r * self.r_gain, g * self.g_gain, b * self.b_gain

        th, ts, _ = colorsys.rgb_to_hsv(
            _clamp01(r / 255.0), _clamp01(g / 255.0), _clamp01(b / 255.0)
        )
        if self.sat_gain > 1.0 and 0.04 < ts < 0.92:
            ts = min(1.0, ts + (1.0 - ts) * (self.sat_gain - 1.0) * 0.45)

        # --- Target room brightness from scene luminance (not RGB magnitude) ---
        scene_l = float(sample.luma)
        # Mix a touch of highlight energy into brightness (lightning, explosions)
        if sample.highlight_luma > scene_l:
            scene_l = 0.82 * scene_l + 0.18 * sample.highlight_luma

        # Soft black knee: night scenes go truly dark
        if sample.black or scene_l < self.black_luma:
            target_bri = self.black_lift * self.room_scale
        else:
            # Perceptual drive (sRGB OETF) so midtone movie scenes aren't crushed
            norm = _compress_luma(scene_l, self.black_luma)
            target_bri = _clamp01(norm * self.room_scale)
            target_bri = max(target_bri, self.min_brightness * self.room_scale)

        # --- Scene cut detection ---
        chroma_now = _rgb_chroma(r, g, b)
        chroma_delta = sum(abs(a - b) for a, b in zip(chroma_now, self._prev_chroma, strict=True))
        luma_delta = abs(scene_l - self._prev_luma)
        is_cut = chroma_delta > 0.55 or luma_delta > 0.22
        self._prev_chroma = chroma_now
        self._prev_luma = scene_l

        if not self._primed:
            self._h, self._s, self._bri = th, ts, target_bri
            self._primed = True
            return self._emit(sample, cut=False)

        # Asymmetric temporal model (physical light)
        if is_cut:
            color_tau = min(self.color_s, 0.05)
            bri_tau = min(self.attack_s, 0.04)
        elif target_bri > self._bri:
            # Attack — flashes / explosions come up fast
            bri_tau = self.attack_s
            color_tau = self.color_s * 0.85
        else:
            # Release — settle into darkness slowly
            bri_tau = self.release_s
            color_tau = self.color_s

        a_c = 1.0 - math.exp(-dt / max(color_tau, 1e-3))
        a_b = 1.0 - math.exp(-dt / max(bri_tau, 1e-3))
        self._h = _lerp_hue(self._h, th, a_c)
        self._s += (ts - self._s) * a_c
        self._bri += (target_bri - self._bri) * a_b

        return self._emit(sample, cut=is_cut)

    def _emit(self, sample: SampleResult, *, cut: bool) -> MovieFrame:
        bri = _clamp01(self._bri)
        # Near-black: kill chroma so we don't get dim gray-blue night glow
        sat = _clamp01(self._s)
        if bri < 0.03:
            sat *= bri / 0.03
        nr, ng, nb = colorsys.hsv_to_rgb(self._h % 1.0, sat, bri)
        out = (
            int(round(nr * 255.0)),
            int(round(ng * 255.0)),
            int(round(nb * 255.0)),
        )
        return MovieFrame(
            rgb=out,
            luma=sample.luma,
            saturation=sample.saturation,
            intensity=self.intensity_name,
            cut=cut,
        )


def _compress_luma(luma: float, black: float) -> float:
    """Map scene linear luma → 0–1 room drive with soft toe/shoulder."""
    x = max(0.0, (luma - black) / max(1e-6, 1.0 - black))
    # Perceptual midtones (movie sRGB content is dark in linear light)
    perc = linear_to_srgb(_clamp01(x))
    # Shoulder so full-bright day still leaves headroom vs the monitor
    return _clamp01(1.0 - math.exp(-2.8 * (perc ** 0.90)))


def _rgb_chroma(r: float, g: float, b: float) -> tuple[float, float, float]:
    lin = (srgb_to_linear(r / 255.0), srgb_to_linear(g / 255.0), srgb_to_linear(b / 255.0))
    s = sum(lin) + 1e-8
    return (lin[0] / s, lin[1] / s, lin[2] / s)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _lerp_hue(a: float, b: float, t: float) -> float:
    d = (b - a + 0.5) % 1.0 - 0.5
    return (a + d * _clamp01(t)) % 1.0
