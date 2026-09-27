from __future__ import annotations

from dataclasses import dataclass
import time

from rgb_lights.lighting.palettes import (
    DARKSIDE_STATE_PALETTE_PREFERENCE,
    STATE_PALETTE_PREFERENCE,
    Palette,
    get_palette,
)
from rgb_lights.music.state import MusicContext, MusicState


@dataclass
class LightIntent:
    palette: Palette
    base_brightness: float
    beat_strength: float
    downbeat_strength: float
    anim_speed: float
    hue_mix: float
    color_step: int  # 0=hold, 1=beat shift, 2=downbeat/accent shift
    accent: bool
    accent_warm: bool
    label: str
    kick_scale: float = 1.0
    tension_drive: float = 0.0
    hush: float = 0.0


class LightingDirector:
    """Map song structure to lighting; color and punch land on the beat grid."""

    def __init__(self, intensity: float = 0.85, style: str = "party") -> None:
        self.style = style
        self.darkside = style == "darkside"
        if self.darkside:
            scale = 0.82
            self._palette_name = "void"
            self._palette_hold_s = 42.0
            self._hue_mix = 0.38
        else:
            scale = 1.0
            self._palette_name = "club"
            self._palette_hold_s = 14.0
            self._hue_mix = 0.45
        self.intensity = max(0.15, min(1.0, intensity * scale))
        self._palette_since = time.perf_counter()
        self._last_state: MusicState | None = None

    def update(self, ctx: MusicContext) -> LightIntent:
        now = time.perf_counter()
        self._maybe_switch_palette(ctx, now)

        palette = get_palette(self._palette_name)
        if self.darkside:
            self._hue_mix = 0.92 * self._hue_mix + 0.08 * (0.22 + ctx.energy * 0.48)
        else:
            self._hue_mix = 0.90 * self._hue_mix + 0.10 * (0.35 + ctx.energy * 0.55)

        base, beat_s, down_s, speed = self._params_for_state(ctx)

        color_step = 0
        if ctx.beat:
            if self.darkside:
                color_step = 2 if ctx.downbeat else (
                    1
                    if ctx.state
                    in (MusicState.ENERGETIC, MusicState.PEAK, MusicState.GROOVE)
                    else 0
                )
            else:
                color_step = 2 if ctx.downbeat else 1
                if ctx.state in (MusicState.CHILL, MusicState.BREAKDOWN) and not ctx.downbeat:
                    color_step = 1

        accent = False
        accent_warm = False
        if ctx.drop_event:
            if self.darkside:
                accent = True
                accent_warm = False
                color_step = 2
                self._palette_name = "eclipse"
                self._palette_since = now
                palette = get_palette(self._palette_name)
                beat_s = max(beat_s, 0.28)
                down_s = max(down_s, 0.40)
                base = max(base, 0.55)
            else:
                accent = True
                accent_warm = True
                color_step = 2
                self._palette_name = "hype"
                self._palette_since = now
                palette = get_palette(self._palette_name)
                beat_s = max(beat_s, 0.42)
                down_s = max(down_s, 0.55)
                base = max(base, 0.82)

        hush = 0.0
        if ctx.pre_drop and not ctx.drop_event:
            if self.darkside:
                hush = 0.16 + ctx.tension * 0.14
                base = max(0.10, base - hush * 0.85)
                beat_s *= 0.60
                speed = min(speed + 0.12, 0.55)
            else:
                hush = 0.22 + ctx.tension * 0.12
                base = max(0.22, base - hush)
                beat_s *= 0.55
                speed = min(speed + 0.25, 1.0)

        if self.darkside:
            kick_scale = 0.48 + ctx.kick_impact * 0.40
            if ctx.state in (MusicState.CHILL, MusicState.BREAKDOWN):
                kick_scale *= 0.85
            elif ctx.state == MusicState.PEAK:
                kick_scale *= 1.08
        else:
            kick_scale = 0.85 + ctx.kick_impact * 0.85
            if ctx.state == MusicState.PEAK:
                kick_scale *= 1.2
            elif ctx.state == MusicState.BUILDUP:
                kick_scale *= 0.7 + 0.5 * ctx.tension
            elif ctx.bpm >= 78:
                kick_scale *= 1.05

        base = _clamp01(base * self.intensity)
        floor = 0.04 * self.intensity if self.darkside else 0.12 * self.intensity
        base = max(base, floor)

        if self.darkside:
            tension_drive = (
                ctx.tension * 0.85
                if ctx.state == MusicState.BUILDUP
                else ctx.tension * 0.40
            )
        else:
            tension_drive = (
                ctx.tension if ctx.state == MusicState.BUILDUP else ctx.tension * 0.35
            )

        self._last_state = ctx.state
        return LightIntent(
            palette=palette,
            base_brightness=base,
            beat_strength=beat_s,
            downbeat_strength=down_s,
            anim_speed=speed,
            hue_mix=_clamp01(self._hue_mix),
            color_step=color_step,
            accent=accent,
            accent_warm=accent_warm,
            label=ctx.state.value,
            kick_scale=kick_scale,
            tension_drive=tension_drive,
            hush=hush,
        )

    def _maybe_switch_palette(self, ctx: MusicContext, now: float) -> None:
        if self.darkside:
            prefs_map = DARKSIDE_STATE_PALETTE_PREFERENCE
            default = ("void",)
        else:
            prefs_map = STATE_PALETTE_PREFERENCE
            default = ("club",)
        prefs = prefs_map.get(ctx.state.value, default)
        age = now - self._palette_since
        big_shift = (
            self._last_state is not None
            and ctx.state != self._last_state
            and ctx.state
            in (MusicState.DROP, MusicState.BREAKDOWN, MusicState.PEAK, MusicState.BUILDUP)
        )
        if self.darkside:
            min_age = 22.0
            if age < min_age and not big_shift and not ctx.drop_event:
                return
            if age < self._palette_hold_s and not big_shift and not ctx.drop_event:
                return
        else:
            if age < 8.0 and not big_shift and not ctx.drop_event:
                return
            if age < self._palette_hold_s and not big_shift and not ctx.drop_event:
                return

        for name in prefs:
            if name != self._palette_name or age > self._palette_hold_s:
                self._palette_name = name
                self._palette_since = now
                if self.darkside:
                    self._palette_hold_s = 32.0 + (1.0 - ctx.energy) * 36.0
                else:
                    self._palette_hold_s = 10.0 + (1.0 - ctx.energy) * 14.0
                return

    def _params_for_state(self, ctx: MusicContext) -> tuple[float, float, float, float]:
        e = ctx.energy
        t = ctx.tension
        if self.darkside:
            if ctx.state == MusicState.CHILL:
                return 0.16 + e * 0.12, 0.14, 0.22, 0.12
            if ctx.state == MusicState.BREAKDOWN:
                return 0.12 + e * 0.10, 0.12, 0.18, 0.10
            if ctx.state == MusicState.GROOVE:
                return 0.26 + e * 0.16, 0.22, 0.34, 0.22
            if ctx.state == MusicState.ENERGETIC:
                return 0.34 + e * 0.18, 0.28, 0.42, 0.32
            if ctx.state == MusicState.BUILDUP:
                return (
                    0.18 + t * 0.32 + e * 0.08,
                    0.18 + t * 0.20,
                    0.28 + t * 0.28,
                    0.20 + t * 0.28,
                )
            if ctx.state == MusicState.DROP:
                return 0.58, 0.38, 0.55, 0.40
            if ctx.state == MusicState.PEAK:
                return 0.38 + e * 0.18, 0.30, 0.46, 0.34
            return 0.24, 0.20, 0.30, 0.20

        if ctx.state == MusicState.CHILL:
            return 0.22 + e * 0.14, 0.28, 0.40, 0.22
        if ctx.state == MusicState.BREAKDOWN:
            return 0.18 + e * 0.12, 0.30, 0.42, 0.20
        if ctx.state == MusicState.GROOVE:
            return 0.28 + e * 0.16, 0.48, 0.62, 0.38
        if ctx.state == MusicState.ENERGETIC:
            return 0.34 + e * 0.18, 0.55, 0.72, 0.52
        if ctx.state == MusicState.BUILDUP:
            return (
                0.20 + t * 0.42 + e * 0.08,
                0.35 + t * 0.30,
                0.48 + t * 0.35,
                0.45 + t * 0.40,
            )
        if ctx.state == MusicState.DROP:
            return 0.75, 0.70, 0.90, 0.75
        if ctx.state == MusicState.PEAK:
            return 0.40 + e * 0.22, 0.60, 0.80, 0.62
        return 0.30, 0.48, 0.62, 0.40


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
