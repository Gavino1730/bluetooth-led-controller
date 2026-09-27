from __future__ import annotations

import colorsys
import time

from rgb_lights.audio.analyzer import SpectrumBands


class ColorMapper:
    """Map spectrum bands to RGB depending on reactive mode."""

    def __init__(self, mode: str = "spectrum") -> None:
        self.mode = mode.lower().strip()
        self._hue = 0.12  # start in warm territory
        self._flash = 0.0
        self._beat_count = 0
        self._last_beat_t = 0.0
        self._prev_mid = 0.0
        self._prev_treble = 0.0
        self._pace = 0.45
        self._hat_cooldown = 0.0
        self.urgent = False  # set when a hit should push lights ASAP

    def map(self, bands: SpectrumBands) -> tuple[int, int, int]:
        if self.mode == "bass_pulse":
            return self._bass_pulse(bands)
        if self.mode == "energy":
            return self._energy(bands)
        if self.mode == "party":
            return self._party(bands)
        if self.mode == "festival":
            return self._festival(bands)
        return self._spectrum(bands)

    def _spectrum(self, bands: SpectrumBands) -> tuple[int, int, int]:
        boost = 0.25 + 0.75 * bands.energy
        r = int(255 * _clamp01(bands.bass * boost * 1.2))
        g = int(255 * _clamp01(bands.mid * boost * 1.1))
        b = int(255 * _clamp01(bands.treble * boost * 1.3))
        if r + g + b < 8 and bands.energy > 0.02:
            return _hsv_to_rgb(0.55, 0.7, 0.12 + bands.energy * 0.4)
        return r, g, b

    def _bass_pulse(self, bands: SpectrumBands) -> tuple[int, int, int]:
        if bands.beat:
            self._hue = (self._hue + 0.08) % 1.0
            self._flash = max(self._flash, 0.55 + bands.beat_strength * 0.2)
        self._flash *= 0.86
        value = 0.12 + bands.bass * 0.55 + self._flash * 0.35
        sat = 0.8 + bands.treble * 0.15
        return _hsv_to_rgb(self._hue, sat, value)

    def _energy(self, bands: SpectrumBands) -> tuple[int, int, int]:
        warmth = bands.bass / (bands.bass + bands.treble + 1e-6)
        hue = 0.02 + (1.0 - warmth) * 0.55
        value = 0.08 + bands.energy * 0.75
        if bands.beat:
            value = min(0.95, value + 0.2)
        return _hsv_to_rgb(hue, 0.9, value)

    def _party(self, bands: SpectrumBands) -> tuple[int, int, int]:
        """House-party vibe: colorful, beat-aware, never harsh/white."""
        now = time.perf_counter()

        # Slow ambient drift so color keeps moving between kicks.
        self._hue = (self._hue + 0.003 + bands.mid * 0.004 + bands.treble * 0.003) % 1.0

        if bands.beat:
            self._beat_count += 1
            self._last_beat_t = now
            # Step through a wide palette (full rainbow, golden-angle-ish jumps).
            self._hue = (self._hue + 0.17) % 1.0
            # Soft lift — not a festival strobe.
            self._flash = 0.55 + min(0.25, bands.beat_strength * 0.25)

        age = now - self._last_beat_t if self._last_beat_t else 9.0
        if age < 0.18:
            self._flash = max(self._flash * 0.92, 0.55 - age * 1.2)
        else:
            self._flash *= 0.88

        hue = self._hue
        # Keep saturation rich but not neon-clipped; no desat toward white.
        sat = 0.78 + bands.treble * 0.12

        # Comfortable room glow + gentle beat bump.
        base = 0.22 + bands.energy * 0.22 + bands.bass * 0.18 + bands.mid * 0.1
        beat_lift = self._flash * 0.35
        value = _clamp01(base + beat_lift)
        # Cap peak brightness so it stays house-party, not festival.
        value = min(value, 0.88)

        return _hsv_to_rgb(hue, sat, value)

    def _festival(self, bands: SpectrumBands) -> tuple[int, int, int]:
        """Festival: pace follows the song; mids = melody, highs = hats."""
        now = time.perf_counter()
        self.urgent = False

        # Don't invent a fake 110 BPM groove when the track is paused / near-silent.
        if bands.energy < 0.04 and bands.bass < 0.04 and bands.treble < 0.05 and not bands.beat:
            self._pace *= 0.78
            self._flash *= 0.65
            self._prev_mid = bands.mid
            self._prev_treble = bands.treble
            sat = 0.75
            value = _clamp01(0.04 + self._flash * 0.12)
            return _hsv_to_rgb(self._hue, sat, value)

        bpm = bands.bpm if bands.bpm > 40.0 else 0.0
        bpm_pace = _clamp01((bpm - 72.0) / 78.0) if bpm > 0 else 0.0
        activity = _clamp01(0.30 * bands.mid + 0.45 * bands.treble + 0.25 * bands.flux)
        target_pace = _clamp01(0.22 * bpm_pace + 0.48 * (bands.energy**0.85) + 0.30 * activity)
        self._pace = 0.84 * self._pace + 0.16 * target_pace
        pace = self._pace

        # Melody (mids): steer hue along the contour instead of random spin.
        mid_delta = bands.mid - self._prev_mid
        self._prev_mid = bands.mid
        melody = (0.52 + bands.mid * 0.58 + mid_delta * 0.40) % 1.0
        melody_blend = (0.035 + 0.11 * pace) * (0.30 + bands.mid * 0.70)
        self._hue = _lerp_hue(self._hue, melody, melody_blend)

        # Drift scales with pace — chill sections crawl, drops race.
        drift = 0.0012 + pace * 0.026 + bands.mid * pace * 0.008
        self._hue = (self._hue + drift) % 1.0

        # Hi-hats: sharp treble rises with highs leading the low end.
        treble_rise = bands.treble - self._prev_treble
        self._prev_treble = bands.treble
        hat_gap = 0.042 if pace > 0.55 else 0.085
        is_hat = (
            treble_rise > 0.055
            and bands.treble > 0.26
            and bands.treble >= bands.bass * 0.80
            and (now - self._hat_cooldown) > hat_gap
        )
        if is_hat:
            self._hat_cooldown = now
            self.urgent = True
            self._hue = (self._hue + 0.035 + pace * 0.055 + treble_rise * 0.10) % 1.0
            self._flash = max(self._flash, 0.16 + bands.treble * 0.38 * (0.45 + pace * 0.55))

        if bands.beat:
            self._beat_count += 1
            self._last_beat_t = now
            self.urgent = True
            # Soft steps when calm; big rainbow leaps when the track is cooking.
            jump = 0.05 + pace * 0.30
            if self._beat_count % 4 == 0:
                jump += 0.05 * pace
            jump *= 0.65 + 0.55 * bands.beat_strength
            self._hue = (self._hue + jump) % 1.0
            self._flash = 0.32 + pace * 0.48 + min(0.20, bands.beat_strength * 0.25)

        if bands.flux > 0.48 and pace > 0.42 and (now - self._last_beat_t) > 0.08:
            self._hue = (self._hue + (0.06 + 0.10 * pace) * bands.flux) % 1.0
            self._flash = max(self._flash, 0.22 + bands.flux * 0.25 * pace)

        self._flash *= 0.90 - pace * 0.14

        sat = min(1.0, 0.80 + bands.treble * 0.16 + pace * 0.06)
        base = (
            0.14
            + bands.energy * 0.30
            + bands.bass * 0.20
            + bands.mid * 0.18
            + bands.treble * 0.12
        )
        base *= 0.52 + 0.48 * (0.30 + pace * 0.70)
        value = _clamp01(base + self._flash * (0.26 + pace * 0.24))
        value = min(value, 0.70 + pace * 0.28)

        return _hsv_to_rgb(self._hue, sat, value)


class RateLimiter:
    def __init__(self, hz: float) -> None:
        self.min_interval = 1.0 / max(1.0, hz)
        self._last = 0.0

    def ready(self) -> bool:
        now = time.perf_counter()
        if now - self._last >= self.min_interval:
            self._last = now
            return True
        return False


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb(_clamp01(h) % 1.0, _clamp01(s), _clamp01(v))
    return int(r * 255), int(g * 255), int(b * 255)


def _lerp_hue(a: float, b: float, t: float) -> float:
    d = (b - a + 0.5) % 1.0 - 0.5
    return (a + d * _clamp01(t)) % 1.0


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
