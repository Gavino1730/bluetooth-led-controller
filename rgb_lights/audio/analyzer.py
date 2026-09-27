from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import time

import numpy as np


@dataclass
class AudioFeatures:
    """Normalized, AGC'd musical features for one analysis frame."""

    rms: float = 0.0
    energy: float = 0.0
    sub: float = 0.0
    bass: float = 0.0
    mid: float = 0.0
    high: float = 0.0
    # Sax / warm synth body (fundamentals + low harmonics)
    body: float = 0.0
    # Sax bite / synth lead / filter (upper mids)
    lead: float = 0.0
    flux: float = 0.0
    onset: bool = False
    onset_strength: float = 0.0
    bpm: float = 0.0
    beat: bool = False
    beat_phase: float = 0.0  # 0..1 within current beat
    downbeat: bool = False  # beat 1 of estimated 4/4 bar
    bar_beat: int = 0  # 0..3
    confidence: float = 0.0


# Back-compat alias used by older reactive modes
@dataclass
class SpectrumBands:
    bass: float
    mid: float
    treble: float
    energy: float
    beat: bool = False
    beat_strength: float = 0.0
    flux: float = 0.0
    bpm: float = 0.0

    @classmethod
    def from_features(cls, f: AudioFeatures) -> "SpectrumBands":
        return cls(
            bass=f.bass,
            mid=f.mid,
            treble=f.high,
            energy=f.energy,
            beat=f.beat,
            beat_strength=f.onset_strength,
            flux=f.flux,
            bpm=f.bpm,
        )


class BeatTracker:
    """Stable BPM metronome: onsets estimate tempo; grid fires the beats."""

    def __init__(
        self,
        sensitivity: float = 1.32,
        min_interval_s: float = 0.22,
        history: int = 56,
    ) -> None:
        self.sensitivity = sensitivity
        self.min_interval_s = min_interval_s
        self._prev_bass = 0.0
        self._flux_hist: deque[float] = deque(maxlen=history)
        self._last_onset_t = 0.0
        self._last_beat_t = 0.0
        self._ibi_hist: deque[float] = deque(maxlen=24)
        self._locked_bpm = 0.0
        self._lock_frames = 0
        self._silent_frames = 0
        self.bpm = 0.0
        self.beat_phase = 0.0
        self.bar_beat = 0
        self.confidence = 0.0

    def update(self, bass: float, now: float) -> tuple[bool, float, bool, bool]:
        flux = max(0.0, bass - self._prev_bass)
        self._prev_bass = bass
        self._flux_hist.append(flux)

        # Pause / silence: stop inventing metronome beats and let the lock fade.
        if bass < 0.045 and flux < 0.008:
            self._silent_frames += 1
        else:
            self._silent_frames = 0

        if self._silent_frames >= 6:
            self.confidence *= 0.88
            if self.confidence < 0.35:
                self.bpm = 0.0
                self._locked_bpm = 0.0
                self._lock_frames = 0
                self.beat_phase = 0.0
            return False, 0.0, False, False

        onset = False
        strength = 0.0
        if len(self._flux_hist) >= 8:
            mean_flux = float(np.mean(self._flux_hist))
            threshold = mean_flux * self.sensitivity + 0.014
            if (
                flux > threshold
                and bass > 0.10
                and (now - self._last_onset_t) >= self.min_interval_s
            ):
                onset = True
                strength = _clamp01(
                    (flux - threshold) / (threshold + 1e-6) * 0.40 + bass * 0.60
                )
                ibi = now - self._last_onset_t
                self._last_onset_t = now
                if 0.28 <= ibi <= 1.2:
                    self._ibi_hist.append(ibi)
                    if len(self._ibi_hist) >= 5:
                        self._lock_bpm()

        locked = self.bpm >= 78 and self.confidence >= 0.40
        beat = False
        downbeat = False

        if locked:
            period = 60.0 / self.bpm
            if self._last_beat_t <= 0:
                self._last_beat_t = now

            # Onsets only nudge phase when they land near a grid beat — never
            # invent extra beats (that was flipping tempo / missing the pulse).
            if onset and self._last_beat_t > 0:
                err = (now - self._last_beat_t) % period
                if err > period * 0.5:
                    err -= period
                if abs(err) <= period * 0.20:
                    # Soft pull toward the real kick
                    self._last_beat_t += err * 0.65
                    strength = max(strength, 0.35)

            age = now - self._last_beat_t
            # Catch up at most one grid tick per frame
            if age >= period * 0.98:
                self._last_beat_t += period
                self.bar_beat = (self.bar_beat + 1) % 4
                beat = True
                downbeat = self.bar_beat == 0
                if not onset:
                    strength = max(strength, 0.32 + 0.25 * self.confidence)
                age = now - self._last_beat_t
            self.beat_phase = _clamp01(age / period)
        else:
            # Before lock: provisional beats from clear onsets only
            if onset and strength > 0.28:
                beat = True
                self._last_beat_t = now
                self.bar_beat = (self.bar_beat + 1) % 4
                downbeat = self.bar_beat == 0
                self.beat_phase = 0.0
            elif self.bpm >= 70 and self._last_beat_t > 0:
                period = 60.0 / self.bpm
                age = now - self._last_beat_t
                self.beat_phase = _clamp01(age / period)

        return onset, strength, beat, downbeat

    def _lock_bpm(self) -> None:
        ibis = np.asarray(self._ibi_hist, dtype=np.float64)
        # Reject outlier intervals so snare/hat chatter doesn't yank tempo
        med = float(np.median(ibis))
        tight = ibis[(ibis > med * 0.72) & (ibis < med * 1.28)]
        if len(tight) < 4:
            tight = ibis
        median_ibi = float(np.median(tight))
        bpm = self._fold_dance_bpm(60.0 / max(median_ibi, 1e-6))

        std = float(np.std(tight))
        consistency = _clamp01(1.0 - (std / (median_ibi + 1e-6)) * 1.4)

        if self._locked_bpm > 0 and self.confidence > 0.50:
            # Stay on the current pulse; only accept near matches or exact half/double
            candidate = self._nearest_pulse(bpm, self._locked_bpm)
            ratio = candidate / self._locked_bpm
            if 0.92 <= ratio <= 1.08:
                self._locked_bpm = 0.88 * self._locked_bpm + 0.12 * candidate
                self._lock_frames += 1
            elif consistency > 0.70 and self._lock_frames > 40:
                # Rare re-lock only when new estimate is very stable
                self._locked_bpm = 0.7 * self._locked_bpm + 0.3 * candidate
            else:
                # Ignore noisy estimate — keep locked tempo
                pass
        else:
            if self._locked_bpm > 0:
                bpm = self._nearest_pulse(bpm, self._locked_bpm)
                self._locked_bpm = 0.65 * self._locked_bpm + 0.35 * bpm
            else:
                self._locked_bpm = bpm
            self._lock_frames += 1

        self.bpm = self._locked_bpm
        self.confidence = 0.90 * self.confidence + 0.10 * max(consistency, 0.35)

    @staticmethod
    def _fold_dance_bpm(bpm: float) -> float:
        while bpm < 84 and bpm > 0:
            bpm *= 2.0
        while bpm > 168:
            bpm *= 0.5
        # Prefer ~100–140 club pulse when ambiguous near edges
        if bpm < 92:
            bpm *= 2.0
        if bpm > 152:
            bpm *= 0.5
        return float(bpm)

    @staticmethod
    def _nearest_pulse(candidate: float, locked: float) -> float:
        """Map candidate onto locked tempo's pulse family (1x / 2x / ½x)."""
        options = [candidate, candidate * 2.0, candidate * 0.5]
        best = min(options, key=lambda x: abs(x - locked) / locked)
        if abs(best - locked) / locked < 0.18:
            return best
        return locked


class AudioAnalyzer:
    """Real-time FFT features with relative AGC for dorm-party lighting."""

    def __init__(self, sample_rate: int = 44100, gain: float = 1.4, smoothing: float = 0.45) -> None:
        self.sample_rate = sample_rate
        self.gain = gain
        self.smoothing = max(0.0, min(0.95, smoothing))
        self._window: np.ndarray | None = None
        self._agc = 1.0
        self._smooth = AudioFeatures()
        self._beats = BeatTracker()
        self._onset_density: deque[float] = deque(maxlen=64)
        self._energy_hist: deque[float] = deque(maxlen=120)

    def analyze(self, samples: np.ndarray, sample_rate: int | None = None) -> AudioFeatures:
        if sample_rate is not None:
            self.sample_rate = sample_rate
        if samples.size == 0:
            return self._smooth

        now = time.perf_counter()
        mono = np.asarray(samples, dtype=np.float32)
        if self._window is None or len(self._window) != len(mono):
            self._window = np.hanning(len(mono)).astype(np.float32)

        windowed = mono * self._window
        power = np.abs(np.fft.rfft(windowed)) ** 2
        freqs = np.fft.rfftfreq(len(windowed), d=1.0 / self.sample_rate)

        sub = self._band_power(power, freqs, 20, 80)
        bass = self._band_power(power, freqs, 80, 200)
        mid = self._band_power(power, freqs, 200, 2000)
        high = self._band_power(power, freqs, 2000, 12000)
        # Sax / warm pad body vs sax bite / synth lead / filter
        body = self._band_power(power, freqs, 250, 900)
        lead = self._band_power(power, freqs, 900, 3200)
        rms = float(np.sqrt(np.mean(mono * mono)))

        # Absolute digital silence (pause): don't let AGC climb and invent music from noise.
        if rms < 0.004:
            self._agc = 0.95 * self._agc + 0.05 * 1.0
            self._energy_hist.append(0.0)
            self._onset_density.append(0.0)
            silent = AudioFeatures(bpm=self._beats.bpm, confidence=self._beats.confidence * 0.9)
            onset, strength, beat, downbeat = self._unpack_beat(self._beats.update(0.0, now))
            silent.onset = onset
            silent.onset_strength = strength
            silent.beat = beat
            silent.downbeat = downbeat
            silent.beat_phase = self._beats.beat_phase
            silent.bar_beat = self._beats.bar_beat
            silent.bpm = self._beats.bpm
            silent.confidence = self._beats.confidence
            self._smooth = AudioFeatures(
                rms=self._lerp(self._smooth.rms, 0.0, 0.15),
                energy=self._lerp(self._smooth.energy, 0.0, 0.15),
                sub=self._lerp(self._smooth.sub, 0.0, 0.12),
                bass=self._lerp(self._smooth.bass, 0.0, 0.12),
                mid=self._lerp(self._smooth.mid, 0.0, 0.15),
                high=self._lerp(self._smooth.high, 0.0, 0.15),
                body=self._lerp(self._smooth.body, 0.0, 0.15),
                lead=self._lerp(self._smooth.lead, 0.0, 0.15),
                flux=0.0,
                onset=silent.onset,
                onset_strength=silent.onset_strength,
                bpm=silent.bpm,
                beat=silent.beat,
                beat_phase=silent.beat_phase,
                downbeat=silent.downbeat,
                bar_beat=silent.bar_beat,
                confidence=silent.confidence,
            )
            return self._smooth

        peak = max(sub, bass, mid, high, body, lead, rms * 4.0, 1e-6)
        self._agc = 0.97 * self._agc + 0.03 * (0.55 / peak)
        self._agc = float(np.clip(self._agc, 0.35, 8.0))
        g = self.gain * self._agc

        raw = AudioFeatures(
            rms=_clamp01(rms * g * 4.0),
            energy=_clamp01(rms * g * 4.0),
            sub=_clamp01(sub * g),
            bass=_clamp01(bass * g),
            mid=_clamp01(mid * g),
            high=_clamp01(high * g),
            body=_clamp01(body * g),
            lead=_clamp01(lead * g),
        )

        # Beat tracker uses combined low end (sub+bass).
        low = _clamp01(0.55 * raw.sub + 0.45 * raw.bass)
        onset, strength, beat, downbeat = self._unpack_beat(self._beats.update(low, now))

        raw.flux = _clamp01(max(0.0, low - (0.55 * self._smooth.sub + 0.45 * self._smooth.bass)) * 5.0)
        raw.onset = onset
        raw.onset_strength = strength
        raw.bpm = self._beats.bpm
        raw.beat = beat
        raw.beat_phase = self._beats.beat_phase
        raw.downbeat = downbeat
        raw.bar_beat = self._beats.bar_beat
        raw.confidence = self._beats.confidence

        self._onset_density.append(1.0 if onset else 0.0)
        self._energy_hist.append(raw.energy)

        s = self.smoothing
        # Body/lead: slightly snappier than mid so sax/synth lines read through
        self._smooth = AudioFeatures(
            rms=self._lerp(self._smooth.rms, raw.rms, s),
            energy=self._lerp(self._smooth.energy, raw.energy, s),
            sub=self._lerp(self._smooth.sub, raw.sub, s * 0.7),
            bass=self._lerp(self._smooth.bass, raw.bass, s * 0.7),
            mid=self._lerp(self._smooth.mid, raw.mid, s),
            high=self._lerp(self._smooth.high, raw.high, s),
            body=self._lerp(self._smooth.body, raw.body, s * 0.85),
            lead=self._lerp(self._smooth.lead, raw.lead, s * 0.80),
            flux=raw.flux,
            onset=raw.onset,
            onset_strength=raw.onset_strength,
            bpm=raw.bpm,
            beat=raw.beat,
            beat_phase=raw.beat_phase,
            downbeat=raw.downbeat,
            bar_beat=raw.bar_beat,
            confidence=raw.confidence,
        )
        return self._smooth

    def recent_onset_density(self) -> float:
        if not self._onset_density:
            return 0.0
        return float(np.mean(self._onset_density))

    def energy_slope(self) -> float:
        """Positive when energy has been climbing over ~recent history."""
        if len(self._energy_hist) < 20:
            return 0.0
        arr = np.asarray(self._energy_hist, dtype=np.float32)
        early = float(np.mean(arr[: len(arr) // 3]))
        late = float(np.mean(arr[-(len(arr) // 3) :]))
        return late - early

    @staticmethod
    def _unpack_beat(result: tuple) -> tuple[bool, float, bool, bool]:
        # BeatTracker.update returns onset, strength, beat, downbeat
        return result[0], result[1], result[2], result[3]

    @staticmethod
    def _band_power(spectrum: np.ndarray, freqs: np.ndarray, low: float, high: float) -> float:
        mask = (freqs >= low) & (freqs < high)
        if not np.any(mask):
            return 0.0
        power = float(np.sum(spectrum[mask]))
        total = float(np.sum(spectrum) + 1e-9)
        return float(np.log1p(power / total * 50.0) / np.log1p(50.0))

    @staticmethod
    def _lerp(previous: float, current: float, smoothing: float) -> float:
        return previous * smoothing + current * (1.0 - smoothing)


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))
