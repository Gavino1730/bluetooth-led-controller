from __future__ import annotations

"""Calm music lighting — glacial fades, no snaps.

Follows sax / synth via body/lead bands. Color and brightness ease with
multi-second time constants so the room never jumps.
"""

from collections import deque
from dataclasses import dataclass
import colorsys
import math
import time

import numpy as np

from rgb_lights.audio.analyzer import AudioAnalyzer, AudioFeatures
from rgb_lights.lighting.palettes import get_palette


@dataclass
class CalmMood:
    level: float
    warmth: float
    presence: float
    air: float
    space: float
    swell: float
    texture: float
    voice: float
    open: float
    label: str
    palette: str


@dataclass
class CalmFrame:
    rgb: tuple[int, int, int]
    features: AudioFeatures
    mood: CalmMood


# Continuous warm→cool arc (lerp_hue short-path stays smooth along this order)
_HUE_STOPS = (
    0.08,  # amber
    0.04,  # copper
    0.98,  # deep rose
    0.92,  # rose
    0.86,  # magenta
    0.78,  # violet
    0.68,  # indigo
    0.58,  # blue
    0.50,  # teal
)


class CalmEngine:
    """Audio → mood → vivid wash with multi-second fades only."""

    # Time constants (seconds) — higher = slower melt
    _TAU_MOOD = 3.5
    _TAU_COLOR_POS = 9.0
    _TAU_HUE = 10.0
    _TAU_SAT = 5.0
    _TAU_VAL = 6.0
    _TAU_RGB = 4.5  # final float-RGB ease (kills integer snaps)
    _MAX_RGB_STEP = 1.8  # hard cap per channel per frame

    def __init__(self, intensity: float = 0.85, gain: float = 1.8, smoothing: float = 0.55) -> None:
        self.intensity = max(0.2, min(1.0, intensity))
        self.analyzer = AudioAnalyzer(gain=gain, smoothing=max(0.70, smoothing))

        self._level = 0.0
        self._warmth = 0.55
        self._presence = 0.0
        self._air = 0.0
        self._texture = 0.0
        self._swell = 0.0
        self._space = 0.6
        self._voice = 0.0
        self._open = 0.35
        self._color_pos = 0.30

        self._prev_body = 0.0
        self._prev_lead = 0.0
        self._hue = 0.09
        self._sat = 1.0
        self._val = 0.30
        self._r = 40.0
        self._g = 18.0
        self._b = 0.0

        self._palette = "amber"
        self._prev_palette = "amber"
        self._palette_mix = 1.0
        self._palette_since = time.perf_counter()
        self._climate_streak = 0
        self._label = "haze"
        self._started = time.perf_counter()
        self._last_t = time.perf_counter()

        self._level_hist: deque[float] = deque(maxlen=96)
        self._voice_hist: deque[float] = deque(maxlen=64)

    def process(self, samples, sample_rate: int) -> CalmFrame:
        now = time.perf_counter()
        dt = max(0.008, min(0.12, now - self._last_t))
        self._last_t = now

        features = self.analyzer.analyze(samples, sample_rate)
        mood = self._update_mood(features, now, dt)
        rgb = self._render(mood, dt)
        return CalmFrame(rgb=rgb, features=features, mood=mood)

    def _update_mood(self, f: AudioFeatures, now: float, dt: float) -> CalmMood:
        a = _alpha(dt, self._TAU_MOOD)

        voice_raw = _clamp01(0.55 * f.body + 0.45 * f.lead)
        self._voice = _exp_smooth(self._voice, voice_raw, a)
        self._voice_hist.append(self._voice)

        open_raw = f.lead / (f.body + f.lead + 0.08)
        self._open = _exp_smooth(self._open, open_raw, a)

        body_d = f.body - self._prev_body
        lead_d = f.lead - self._prev_lead
        self._prev_body = f.body
        self._prev_lead = f.lead

        self._presence = _exp_smooth(self._presence, voice_raw, a)

        tex_raw = _clamp01(abs(body_d) * 2.2 + abs(lead_d) * 2.4)
        self._texture = _exp_smooth(self._texture, tex_raw, _alpha(dt, self._TAU_MOOD * 1.2))

        raw_level = _clamp01(0.40 * f.energy + 0.40 * voice_raw + 0.12 * f.bass + 0.08 * f.mid)
        self._level = _exp_smooth(self._level, raw_level, a)
        self._level_hist.append(self._level)

        lows = f.sub * 0.50 + f.bass * 0.70 + f.body * 0.35
        highs = f.high * 1.00 + f.lead * 0.25
        warm_raw = _clamp01(lows / (lows + highs + 0.08) - f.high * 0.08)
        self._warmth = _exp_smooth(self._warmth, warm_raw, _alpha(dt, self._TAU_MOOD * 1.4))

        self._air = _exp_smooth(self._air, f.high, a)

        activity = _clamp01(self._level * 1.05 + self._voice * 0.70 + self._air * 0.18)
        self._space = _exp_smooth(self._space, _clamp01(1.0 - activity), a)

        slope = self._slow_slope(self._level_hist)
        voice_slope = self._slow_slope(self._voice_hist)
        booting = (now - self._started) < 3.0 or len(self._level_hist) < 48
        if not booting and (slope > 0.010 or voice_slope > 0.012) and self._level > 0.08:
            self._swell = min(1.0, self._swell + dt * (0.08 + max(slope, voice_slope) * 4.0))
        else:
            self._swell = max(0.0, self._swell * math.exp(-dt / 4.0) - dt * 0.02)

        target_pos = _clamp01(
            0.08
            + (1.0 - self._warmth) * 0.50
            + self._open * 0.38
            + self._voice * 0.12
            + self._air * 0.18
        )
        self._color_pos = _exp_smooth(
            self._color_pos, target_pos, _alpha(dt, self._TAU_COLOR_POS)
        )

        self._pick_palette(now, dt)
        self._label = self._pick_label()

        return CalmMood(
            level=self._level,
            warmth=self._warmth,
            presence=self._presence,
            air=self._air,
            space=self._space,
            swell=self._swell,
            texture=self._texture,
            voice=self._voice,
            open=self._open,
            label=self._label,
            palette=self._palette,
        )

    def _pick_label(self) -> str:
        if self._space > 0.72 or self._level < 0.06:
            return "space"
        if self._swell > 0.45:
            return "swell"
        if self._voice > 0.38 and self._level > 0.18:
            return "bloom"
        if self._warmth > 0.62:
            return "glow"
        return "haze"

    def _pick_palette(self, now: float, dt: float) -> None:
        target = self._palette_for_climate()
        # ~15–20s to finish a palette crossfade
        self._palette_mix = min(1.0, self._palette_mix + dt / 16.0)

        if target == self._palette:
            self._climate_streak = 0
            return

        self._climate_streak += 1
        age = now - self._palette_since
        ready = self._climate_streak >= 40 and age > 12.0
        first = age > 5.0 and self._climate_streak >= 24
        if ready or first:
            self._prev_palette = self._palette
            self._palette = target
            self._palette_mix = 0.0
            self._palette_since = now
            self._climate_streak = 0

    def _palette_for_climate(self) -> str:
        w, p, a, o = self._warmth, self._presence, self._air, self._open
        if self._space > 0.65:
            return "fog" if a > 0.25 else "indigo"
        if p > 0.32 and o > 0.55:
            return "velvet" if w < 0.50 else "dusk"
        if p > 0.36:
            return "velvet" if w < 0.55 else "dusk"
        if w > 0.68:
            return "amber"
        if w < 0.38:
            return "indigo" if a < 0.35 else "fog"
        if w > 0.52:
            return "dusk"
        return "fog"

    def _render(self, mood: CalmMood, dt: float) -> tuple[int, int, int]:
        target_h = self._mood_hue(mood)

        # Glacial hue / sat / val — time-constant eased
        self._hue = _lerp_hue(self._hue, target_h, _alpha(dt, self._TAU_HUE))
        self._sat = _exp_smooth(self._sat, 1.0, _alpha(dt, self._TAU_SAT))

        base = 0.16 + mood.level * 0.30 + mood.voice * 0.26 + mood.presence * 0.08
        base += mood.swell * 0.10
        base -= mood.space * 0.12
        # Very slow breath (~25s period) — not a visible pulse
        breath = 0.5 + 0.5 * math.sin(time.perf_counter() * 0.25)
        base += breath * 0.015 * (1.0 - mood.space)

        brightness = _clamp01(base * self.intensity)
        brightness = min(0.62, brightness)
        if mood.space > 0.75:
            brightness = min(brightness, 0.24)

        self._val = _exp_smooth(self._val, brightness, _alpha(dt, self._TAU_VAL))

        tr, tg, tb = _hsv_to_rgb_f(self._hue, self._sat, self._val)
        tr, tg, tb = _vivid_boost_f(tr, tg, tb, amount=0.28)

        # Final float RGB ease + hard per-frame cap → no visible jumps
        a_rgb = _alpha(dt, self._TAU_RGB)
        self._r = _slew_channel(self._r, tr, a_rgb, self._MAX_RGB_STEP)
        self._g = _slew_channel(self._g, tg, a_rgb, self._MAX_RGB_STEP)
        self._b = _slew_channel(self._b, tb, a_rgb, self._MAX_RGB_STEP)

        return (
            int(round(self._r)),
            int(round(self._g)),
            int(round(self._b)),
        )

    def _mood_hue(self, mood: CalmMood) -> float:
        h = _hue_along_path(self._color_pos)

        old_h = get_palette(self._prev_palette).primary[0]
        new_h = get_palette(self._palette).primary[0]
        if mood.open > 0.5 and mood.voice > 0.25:
            new_h = get_palette(self._palette).accent[0]
            old_h = get_palette(self._prev_palette).accent[0]
        elif mood.voice > 0.35:
            new_h = get_palette(self._palette).secondary[0]
            old_h = get_palette(self._prev_palette).secondary[0]

        pal_h = _lerp_hue(old_h, new_h, self._palette_mix)
        h = _lerp_hue(h, pal_h, 0.10)

        if mood.air > 0.20 or (mood.open > 0.48 and mood.voice > 0.2):
            cool = 0.58 if mood.air > mood.open else 0.78
            h = _lerp_hue(h, cool, 0.12 + mood.open * 0.10 + mood.air * 0.08)

        if mood.swell > 0.25:
            accent = 0.92 if mood.warmth < 0.5 else 0.12
            h = _lerp_hue(h, accent, mood.swell * 0.10)

        return h % 1.0

    @staticmethod
    def _slow_slope(hist: deque[float]) -> float:
        if len(hist) < 24:
            return 0.0
        arr = np.asarray(hist, dtype=np.float32)
        third = max(6, len(arr) // 3)
        early = float(np.mean(arr[:third]))
        late = float(np.mean(arr[-third:]))
        return late - early


def _alpha(dt: float, tau: float) -> float:
    """1 - e^(-dt/tau) — frame-rate independent ease amount."""
    if tau <= 1e-6:
        return 1.0
    return 1.0 - math.exp(-max(0.0, dt) / tau)


def _exp_smooth(current: float, target: float, alpha: float) -> float:
    return current + (target - current) * _clamp01(alpha)


def _slew_channel(current: float, target: float, alpha: float, max_step: float) -> float:
    desired = current + (target - current) * _clamp01(alpha)
    delta = desired - current
    if delta > max_step:
        delta = max_step
    elif delta < -max_step:
        delta = -max_step
    return max(0.0, min(255.0, current + delta))


def _hue_along_path(pos: float) -> float:
    pos = _clamp01(pos)
    n = len(_HUE_STOPS) - 1
    x = pos * n
    i = int(x)
    if i >= n:
        return _HUE_STOPS[-1]
    t = x - i
    return _lerp_hue(_HUE_STOPS[i], _HUE_STOPS[i + 1], t)


def _hsv_to_rgb_f(h: float, s: float, v: float) -> tuple[float, float, float]:
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, _clamp01(s), _clamp01(v))
    return r * 255.0, g * 255.0, b * 255.0


def _vivid_boost_f(r: float, g: float, b: float, amount: float = 0.28) -> tuple[float, float, float]:
    mx = max(r, g, b)
    if mx < 8.0:
        return r, g, b
    avg = (r + g + b) / 3.0
    rr = avg + (r - avg) * (1.0 + amount)
    gg = avg + (g - avg) * (1.0 + amount)
    bb = avg + (b - avg) * (1.0 + amount)
    rr = max(0.0, rr)
    gg = max(0.0, gg)
    bb = max(0.0, bb)
    if r >= mx * 0.15:
        rr = max(rr, mx * 0.12 * (r / mx))
    if g >= mx * 0.15:
        gg = max(gg, mx * 0.12 * (g / mx))
    if b >= mx * 0.15:
        bb = max(bb, mx * 0.12 * (b / mx))
    peak = max(rr, gg, bb, 1.0)
    scale = mx / peak
    return rr * scale, gg * scale, bb * scale


def _lerp_hue(a: float, b: float, t: float) -> float:
    d = (b - a + 0.5) % 1.0 - 0.5
    return (a + d * _clamp01(t)) % 1.0


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
