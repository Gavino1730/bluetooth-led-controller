from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import time

import numpy as np

from rgb_lights.audio.analyzer import AudioAnalyzer, AudioFeatures


class MusicState(str, Enum):
    CHILL = "chill"
    GROOVE = "groove"
    ENERGETIC = "energetic"
    BUILDUP = "buildup"
    DROP = "drop"
    BREAKDOWN = "breakdown"
    PEAK = "peak"


@dataclass
class MusicContext:
    state: MusicState
    energy: float
    danceability: float
    bass_energy: float
    mid_energy: float
    high_energy: float
    sub_energy: float
    tension: float
    beat_phase: float
    bpm: float
    beat: bool
    downbeat: bool
    bar_beat: int
    onset_density: float
    energy_slope: float
    drop_event: bool  # True only on the frame a drop is declared
    pre_drop: bool  # Brief hush / riser peak before the hit
    kick_impact: float  # 0–1 how hard this beat should punch lights
    onset_strength: float = 0.0
    flux: float = 0.0


class MusicStateTracker:
    """Classify song structure the way people feel it: build → drop → peak."""

    def __init__(self) -> None:
        self.state = MusicState.CHILL
        self._entered_at = time.perf_counter()
        self._energy = 0.0
        self._dance = 0.0
        self._bass = 0.0
        self._high = 0.0
        self._tension = 0.0
        self._build_score = 0.0
        self._drop_cooldown_until = 0.0
        self._post_drop_until = 0.0
        self._pre_drop = False
        self._pre_drop_until = 0.0

        # Short histories (~2–4s at ~18 Hz) for structure cues
        self._energy_hist: deque[float] = deque(maxlen=72)
        self._bass_hist: deque[float] = deque(maxlen=72)
        self._high_hist: deque[float] = deque(maxlen=72)
        self._density_hist: deque[float] = deque(maxlen=48)

        self._min_dwell = {
            MusicState.CHILL: 2.0,
            MusicState.GROOVE: 1.6,
            MusicState.ENERGETIC: 1.6,
            MusicState.BUILDUP: 0.9,
            MusicState.DROP: 0.4,
            MusicState.BREAKDOWN: 1.8,
            MusicState.PEAK: 2.0,
        }

    def update(self, features: AudioFeatures, analyzer: AudioAnalyzer) -> MusicContext:
        now = time.perf_counter()
        onset_density = analyzer.recent_onset_density()
        slope = analyzer.energy_slope()

        # Smooth meters
        self._energy = 0.88 * self._energy + 0.12 * features.energy
        bass = 0.55 * features.sub + 0.45 * features.bass
        self._bass = 0.82 * self._bass + 0.18 * bass
        self._high = 0.85 * self._high + 0.15 * features.high
        dance_raw = (
            0.40 * self._energy
            + 0.30 * onset_density
            + 0.20 * features.confidence
            + 0.10 * min(1.0, features.bpm / 128.0 if features.bpm > 0 else 0.0)
        )
        self._dance = 0.90 * self._dance + 0.10 * dance_raw

        self._energy_hist.append(self._energy)
        self._bass_hist.append(self._bass)
        self._high_hist.append(self._high)
        self._density_hist.append(onset_density)

        bass_jump = self._relative_jump(self._bass_hist, self._bass)
        high_rise = self._slope_of(self._high_hist)
        energy_dip = self._recent_dip(self._energy_hist)
        density_rise = self._slope_of(self._density_hist)

        # --- Tension / buildup language ---
        # Builds: climbing energy, rising highs (risers), denser hits, less relative bass
        climbing = slope > 0.025 or high_rise > 0.02
        if climbing and self._energy > 0.22:
            lift = 0.028 + max(0.0, slope) * 0.22 + max(0.0, high_rise) * 0.35
            if density_rise > 0.01:
                lift += 0.02
            # Classic pre-drop: highs up while bass holds or dips
            if self._high > self._bass + 0.08:
                lift += 0.025
            self._tension = min(1.0, self._tension + lift)
        elif energy_dip > 0.08 and self._tension > 0.35:
            # Brief hush keeps tension (people lean in)
            self._tension = min(1.0, self._tension + 0.04)
        else:
            decay = 0.012 if self.state == MusicState.BUILDUP else 0.018
            self._tension = max(0.0, self._tension * 0.988 - decay)

        build_raw = _clamp01(
            max(0.0, slope) * 5.0
            + max(0.0, high_rise) * 4.0
            + max(0.0, density_rise) * 3.0
            + self._tension * 0.55
            + (0.15 if self._high > self._bass else 0.0)
            - 0.12
        )
        self._build_score = 0.82 * self._build_score + 0.18 * build_raw

        # Pre-drop: hush or riser peak after sustained tension
        self._pre_drop = False
        if (
            self._tension > 0.55
            and self._build_score > 0.40
            and now >= self._drop_cooldown_until
            and (energy_dip > 0.10 or (self._high > 0.45 and self._bass < 0.35))
        ):
            self._pre_drop = True
            self._pre_drop_until = now + 0.55

        if now < self._pre_drop_until:
            self._pre_drop = True

        drop_event = False
        candidate = self._classify(features, onset_density, slope, now)

        # Drop: hard low-end hit after tension / buildup / pre-drop
        in_build_lane = (
            self.state == MusicState.BUILDUP
            or self._build_score > 0.42
            or self._tension > 0.48
            or self._pre_drop
        )
        strong_hit = (
            features.onset
            and (features.sub > 0.38 or features.bass > 0.42 or bass_jump > 0.18)
            and features.onset_strength > 0.28
        )
        energy_surge = features.energy > self._energy + 0.12 and bass_jump > 0.12

        if (
            now >= self._drop_cooldown_until
            and in_build_lane
            and (strong_hit or energy_surge)
            and (self._tension > 0.40 or self._pre_drop or self._build_score > 0.50)
        ):
            candidate = MusicState.DROP
            drop_event = True
            self._drop_cooldown_until = now + 7.0
            self._post_drop_until = now + 14.0
            self._tension = 0.12
            self._build_score = 0.0
            self._pre_drop = False
            self._pre_drop_until = 0.0

        if candidate != self.state:
            dwell = self._min_dwell.get(self.state, 1.5)
            if self.state == MusicState.DROP:
                dwell = 0.35
            if self.state == MusicState.BUILDUP and candidate in (
                MusicState.DROP,
                MusicState.PEAK,
                MusicState.ENERGETIC,
            ):
                dwell = 0.5
            if (now - self._entered_at) >= dwell or drop_event:
                self.state = candidate
                self._entered_at = now

        # Drop → peak/energetic; stay peak while post-drop heat lasts
        if self.state == MusicState.DROP and (now - self._entered_at) > 0.35:
            self.state = MusicState.PEAK if self._energy > 0.48 else MusicState.ENERGETIC
            self._entered_at = now

        if now < self._post_drop_until and self.state in (
            MusicState.ENERGETIC,
            MusicState.GROOVE,
        ):
            if self._energy > 0.45 and self._dance > 0.35:
                self.state = MusicState.PEAK

        kick_impact = self._kick_impact(features, bass_jump)

        return MusicContext(
            state=self.state,
            energy=self._energy,
            danceability=self._dance,
            bass_energy=self._bass,
            mid_energy=features.mid,
            high_energy=features.high,
            sub_energy=features.sub,
            tension=self._tension,
            beat_phase=features.beat_phase,
            bpm=features.bpm,
            beat=features.beat,
            downbeat=features.downbeat,
            bar_beat=features.bar_beat,
            onset_density=onset_density,
            energy_slope=slope,
            drop_event=drop_event,
            pre_drop=self._pre_drop,
            kick_impact=kick_impact,
            onset_strength=features.onset_strength,
            flux=features.flux,
        )

    def _kick_impact(self, features: AudioFeatures, bass_jump: float) -> float:
        if not features.beat:
            return 0.0
        impact = 0.25 + features.onset_strength * 0.45 + self._bass * 0.25 + bass_jump * 0.35
        if features.downbeat:
            impact += 0.18
        if self.state in (MusicState.PEAK, MusicState.DROP, MusicState.ENERGETIC):
            impact += 0.12
        if self.state == MusicState.BUILDUP:
            impact *= 0.55 + 0.45 * self._tension  # soft early, harder late in build
        if self.state in (MusicState.CHILL, MusicState.BREAKDOWN):
            impact *= 0.45
        return _clamp01(impact)

    def _classify(
        self,
        features: AudioFeatures,
        onset_density: float,
        slope: float,
        now: float,
    ) -> MusicState:
        e = self._energy
        # Prefer staying in buildup while tension is climbing
        if self._build_score > 0.48 and (slope > 0.02 or self._tension > 0.42) and e > 0.24:
            return MusicState.BUILDUP
        if self._tension > 0.62 and e > 0.28 and self._build_score > 0.35:
            return MusicState.BUILDUP
        if e < 0.16 and onset_density < 0.10:
            return MusicState.CHILL
        if e < 0.26 and onset_density < 0.18:
            if self.state in (MusicState.PEAK, MusicState.ENERGETIC, MusicState.DROP):
                return MusicState.BREAKDOWN
            return MusicState.CHILL
        # After a drop, keep peak while still hot
        if now < self._post_drop_until and e > 0.50 and self._dance > 0.38:
            return MusicState.PEAK
        if e > 0.58 and self._dance > 0.42 and self._bass > 0.35:
            return MusicState.PEAK
        if e > 0.40 or onset_density > 0.26:
            return MusicState.ENERGETIC
        return MusicState.GROOVE

    @staticmethod
    def _slope_of(hist: deque[float]) -> float:
        if len(hist) < 12:
            return 0.0
        arr = np.asarray(hist, dtype=np.float32)
        third = max(3, len(arr) // 3)
        early = float(np.mean(arr[:third]))
        late = float(np.mean(arr[-third:]))
        return late - early

    @staticmethod
    def _relative_jump(hist: deque[float], current: float) -> float:
        if len(hist) < 10:
            return 0.0
        arr = np.asarray(hist, dtype=np.float32)
        baseline = float(np.mean(arr[:-3])) if len(arr) > 3 else float(np.mean(arr))
        return max(0.0, current - baseline)

    @staticmethod
    def _recent_dip(hist: deque[float]) -> float:
        """How much current energy sits below the recent peak (hush before drop)."""
        if len(hist) < 16:
            return 0.0
        arr = np.asarray(hist, dtype=np.float32)
        peak = float(np.max(arr[-40:]))
        recent = float(np.mean(arr[-4:]))
        return max(0.0, peak - recent)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
