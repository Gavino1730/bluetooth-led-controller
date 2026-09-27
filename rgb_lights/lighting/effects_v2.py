from __future__ import annotations

import colorsys
import math
import time

from rgb_lights.lighting.chroma import timbre_to_hsv
from rgb_lights.lighting.director import LightIntent
from rgb_lights.music.state import MusicContext


class EffectEngineV2:
    """v2 test: brightness from music structure; color from sound→hue association."""

    def __init__(self, style: str = "party2") -> None:
        self.style = style  # party2 | festival2
        self._hue = 0.08
        self._sat = 0.9
        self._val = 0.45
        self._pulse = 0.0
        self._accent_until = 0.0
        self._bass_smooth = 0.0
        self._energy_smooth = 0.0
        self._timbre_h = 0.08
        self._timbre_s = 0.9
        self._beat_n = 0

    def render(self, intent: LightIntent, ctx: MusicContext) -> tuple[int, int, int]:
        now = time.perf_counter()
        festival = self.style == "festival2"

        # --- Color from timbre (similar sounds → similar colors) ---
        th, ts, hint = timbre_to_hsv(
            sub=ctx.sub_energy,
            bass=ctx.bass_energy,
            mid=ctx.mid_energy,
            high=ctx.high_energy,
            flux=ctx.flux + max(0.0, ctx.energy_slope) * 0.5,
            onset_strength=max(ctx.kick_impact, ctx.onset_strength),
        )

        # Follow timbre faster so band changes show up as color changes
        follow_h = 0.38 if festival else 0.28
        if ctx.beat or ctx.kick_impact > 0.28:
            follow_h = 0.85 if festival else 0.72
        if intent.accent:
            follow_h = 0.95

        self._timbre_h = _lerp_hue(self._timbre_h, th, follow_h)
        self._timbre_s = 0.70 * self._timbre_s + 0.30 * ts

        # On each beat: jump to timbre color + a stepped offset so colors keep moving
        if ctx.beat:
            self._beat_n += 1
            # Stay in the sound's color family, but rotate through nearby hues
            step = 0.14 if festival else 0.10
            if ctx.downbeat:
                step += 0.12 if festival else 0.08
            family = (self._timbre_h + (self._beat_n % 5) * step) % 1.0
            # Also pull toward opposite side a bit on odd beats for bigger swings
            if self._beat_n % 2 == 1:
                family = _lerp_hue(family, (self._timbre_h + 0.35) % 1.0, 0.35 if festival else 0.25)
            self._hue = _lerp_hue(self._hue, family, 0.90 if festival else 0.78)
        else:
            # Between beats still chase live timbre (mids/hats moving)
            self._hue = _lerp_hue(self._hue, self._timbre_h, 0.32 if festival else 0.22)

        self._sat = 0.70 * self._sat + 0.30 * max(0.88, self._timbre_s)

        # Constant gentle churn so color never sits still
        drift = 0.004 + ctx.high_energy * 0.006 + ctx.mid_energy * 0.004
        if festival:
            drift *= 1.6
        self._hue = (self._hue + drift) % 1.0

        if intent.accent:
            self._accent_until = now + (0.35 if festival else 0.45)
            flash_h = _lerp_hue(self._timbre_h, 0.92, 0.55)
            self._hue = flash_h
            self._sat = 0.98
            self._pulse = max(self._pulse, 0.9)

        # --- Brightness: same wide-range idea as v1, style-scaled ---
        self._bass_smooth = 0.40 * self._bass_smooth + 0.60 * ctx.bass_energy
        self._energy_smooth = 0.55 * self._energy_smooth + 0.45 * ctx.energy

        floor = intent.base_brightness * (0.18 if festival else 0.22)
        drive = self._bass_smooth * 0.55 + self._energy_smooth * 0.22 + hint * 0.12

        grid = 0.0
        if ctx.bpm >= 78:
            grid_amt = (0.28 if festival else 0.22) + intent.beat_strength * (0.70 if festival else 0.58)
            if ctx.state.value == "peak":
                grid_amt *= 1.2
            elif ctx.state.value == "buildup":
                grid_amt *= 0.55 + 0.55 * ctx.tension
            elif ctx.state.value in ("chill", "breakdown"):
                grid_amt *= 0.40
            grid = (0.5 + 0.5 * math.cos(ctx.beat_phase * math.tau)) ** 1.75 * grid_amt

        if ctx.beat:
            amount = intent.downbeat_strength if ctx.downbeat else intent.beat_strength
            amount *= intent.kick_scale * (1.25 if festival else 1.1)
            amount *= 0.70 + 0.55 * self._bass_smooth
            self._pulse = max(self._pulse, _clamp01(amount))
        self._pulse *= 0.66 if festival else 0.70

        # Hats / air: short bright ticks in their associated (cool) color
        if festival and ctx.high_energy > 0.35 and ctx.high_energy > ctx.bass_energy * 0.75:
            if ctx.onset_density > 0.15:
                self._pulse = max(self._pulse, 0.18 + ctx.high_energy * 0.25)
                cool = (self._timbre_h + 0.08) % 1.0
                self._hue = _lerp_hue(self._hue, cool, 0.45)
        brightness = _clamp01(floor + drive + grid + self._pulse)
        brightness = _expand_contrast(brightness, 0.10 if festival else 0.12, 1.0)

        if now < self._accent_until:
            t = (self._accent_until - now) / 0.45
            brightness = max(brightness, 0.80 + 0.20 * t)

        if ctx.state.value == "buildup":
            brightness = _clamp01(brightness + ctx.tension * 0.18)
        if intent.hush > 0:
            brightness = _clamp01(brightness * (1.0 - intent.hush * 0.9))
        if ctx.state.value == "peak":
            brightness = _clamp01(brightness + (0.06 if festival else 0.04))

        follow = 0.15 if ctx.beat else 0.30
        self._val = follow * self._val + (1.0 - follow) * brightness
        return _hsv_to_rgb(self._hue, self._sat, self._val)


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, _clamp01(s), _clamp01(v))
    return int(r * 255), int(g * 255), int(b * 255)


def _lerp_hue(a: float, b: float, t: float) -> float:
    d = (b - a + 0.5) % 1.0 - 0.5
    return (a + d * _clamp01(t)) % 1.0


def _expand_contrast(value: float, lo: float, hi: float) -> float:
    v = _clamp01(value)
    curved = v * v * (3.0 - 2.0 * v)
    stretched = curved ** 0.85
    return lo + (hi - lo) * stretched


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
