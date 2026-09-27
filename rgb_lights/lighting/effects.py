from __future__ import annotations

import colorsys
import math
import time

from rgb_lights.lighting.director import LightIntent
from rgb_lights.music.state import MusicContext


class EffectEngine:
    """Brightness and color land on the beat grid so the room can feel the pulse."""

    def __init__(self, style: str = "party") -> None:
        self.style = style
        self._hue = 0.58
        self._sat = 0.85
        self._val = 0.5
        self._pulse = 0.0
        self._accent_until = 0.0
        self._slot = 0
        self._target_h = 0.58
        self._target_s = 0.85
        self._bass_smooth = 0.0
        self._energy_smooth = 0.0
        self._build_spin = 0.0

    def render(self, intent: LightIntent, ctx: MusicContext) -> tuple[int, int, int]:
        now = time.perf_counter()
        colors = (intent.palette.primary, intent.palette.secondary, intent.palette.accent)
        state = ctx.state.value

        # Color moves on beat — pick next palette color; downbeats skip for bigger jumps
        if intent.color_step >= 2:
            self._slot = (self._slot + 2) % 3
            c = colors[self._slot]
            self._target_h, self._target_s = c[0], max(0.90, c[1])
        elif intent.color_step == 1:
            self._slot = (self._slot + 1) % 3
            c = colors[self._slot]
            # Nudge toward a farther hue so consecutive beats don't look similar
            other = colors[(self._slot + 1) % 3]
            spread = 0.35 + intent.hue_mix * 0.40
            self._target_h = _lerp_hue(c[0], other[0], spread * 0.35)
            self._target_s = max(0.88, c[1])

        # Buildup: faster hue crawl across more range
        if state == "buildup" or intent.tension_drive > 0.35:
            self._build_spin = intent.tension_drive * (0.006 + intent.anim_speed * 0.010)
            self._target_h = (self._target_h + self._build_spin) % 1.0
            self._target_s = min(0.98, self._target_s + intent.tension_drive * 0.04)
        else:
            self._build_spin *= 0.9

        # Snap hard on beats so color changes read clearly
        ease = 0.72 if intent.color_step >= 2 else (0.58 if intent.color_step else (0.14 + intent.anim_speed * 0.06))
        if intent.accent:
            ease = 0.80
        self._hue = _lerp_hue(self._hue, self._target_h, ease)
        self._sat = 0.75 * self._sat + 0.25 * max(0.88, self._target_s)

        # Wide dynamic range for dim strips: dark valleys, near-full kicks
        self._bass_smooth = 0.40 * self._bass_smooth + 0.60 * ctx.bass_energy
        self._energy_smooth = 0.55 * self._energy_smooth + 0.45 * ctx.energy

        floor = intent.base_brightness * 0.22
        drive = self._bass_smooth * 0.58 + self._energy_smooth * 0.22

        grid = 0.0
        if ctx.bpm >= 78:
            grid_amt = 0.22 + intent.beat_strength * 0.62
            if state == "peak":
                grid_amt *= 1.25
            elif state == "buildup":
                grid_amt *= 0.60 + 0.55 * ctx.tension
            elif state in ("chill", "breakdown"):
                grid_amt *= 0.45
            phase = ctx.beat_phase
            grid = (0.5 + 0.5 * math.cos(phase * math.tau)) ** 1.8 * grid_amt

        if ctx.beat:
            amount = intent.downbeat_strength if ctx.downbeat else intent.beat_strength
            amount *= intent.kick_scale
            amount *= 0.75 + 0.55 * self._bass_smooth
            if ctx.downbeat:
                amount *= 1.25
            self._pulse = max(self._pulse, _clamp01(amount * 1.15))
        self._pulse *= 0.68

        brightness = _clamp01(floor + drive + grid + self._pulse)
        brightness = _expand_contrast(brightness, 0.12, 1.0)

        if intent.accent:
            self._accent_until = now + 0.40
            target = intent.palette.accent if intent.accent_warm else intent.palette.primary
            self._target_h = target[0]
            self._target_s = max(0.88, target[1])
            self._hue = _lerp_hue(self._hue, self._target_h, 0.75)
            self._pulse = max(self._pulse, 0.85)
            brightness = 1.0

        if now < self._accent_until:
            t = (self._accent_until - now) / 0.40
            brightness = max(brightness, 0.78 + 0.22 * t)
            self._sat = min(0.98, self._sat + 0.02 * t)

        if state == "buildup":
            brightness = _clamp01(brightness + ctx.tension * 0.18)
        if intent.hush > 0:
            brightness = _clamp01(brightness * (1.0 - intent.hush * 0.92))
        if state == "peak":
            brightness = _clamp01(brightness + 0.04)
            self._sat = min(0.97, self._sat + 0.012)

        follow = 0.18 if ctx.beat else 0.32
        self._val = follow * self._val + (1.0 - follow) * brightness
        return _hsv_to_rgb(self._hue, self._sat, self._val)


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, _clamp01(s), _clamp01(v))
    return int(r * 255), int(g * 255), int(b * 255)


def _lerp_hue(a: float, b: float, t: float) -> float:
    d = (b - a + 0.5) % 1.0 - 0.5
    return (a + d * _clamp01(t)) % 1.0


def _expand_contrast(value: float, lo: float, hi: float) -> float:
    """Stretch brightness so dim hardware shows clear dark↔bright swings."""
    v = _clamp01(value)
    curved = v * v * (3.0 - 2.0 * v)
    stretched = curved ** 0.85
    return lo + (hi - lo) * stretched


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
