"""The Dark Side of the Moon lighting — Hipgnosis prism + track-aware movements.

Classifies live audio into album-like sonic regimes and lights them like the
record / 1974 show, while colour + brightness chase whichever voice is loudest:

  drums / kick  → red–orange punch
  guitar        → warm gold–green body (Gilmour / pedal steel / sax)
  synth         → cyan–violet spectral motion (VCS3 / Synthi)
  air / hats    → blue–violet prism edge

Spectrum bands match the cover: ROYGBV (indigo omitted — Hipgnosis).
"""

from __future__ import annotations

from collections import deque
from enum import Enum
import colorsys
import math
import time

from rgb_lights.lighting.director import LightIntent
from rgb_lights.music.state import MusicContext


PRISM_SPECTRUM: tuple[tuple[float, float, float], ...] = (
    (0.00, 0.96, 0.96),   # red
    (0.075, 0.96, 0.96),  # orange
    (0.145, 0.94, 0.96),  # yellow
    (0.33, 0.88, 0.92),   # green
    (0.58, 0.90, 0.95),   # blue
    (0.78, 0.86, 0.94),   # violet
)

WHITE_BEAM = (0.58, 0.06, 0.94)


class AlbumMovement(str, Enum):
    HEARTBEAT = "heartbeat"
    BREATHE = "breathe"
    ON_THE_RUN = "on_the_run"
    TIME = "time"
    GREAT_GIG = "great_gig"
    MONEY = "money"
    US_AND_THEM = "us_and_them"
    ANY_COLOUR = "any_colour"
    BRAIN_DAMAGE = "brain_damage"
    ECLIPSE = "eclipse"


MOVEMENT_LABEL = {
    AlbumMovement.HEARTBEAT: "Speak/Heartbeat",
    AlbumMovement.BREATHE: "Breathe",
    AlbumMovement.ON_THE_RUN: "On the Run",
    AlbumMovement.TIME: "Time",
    AlbumMovement.GREAT_GIG: "Great Gig",
    AlbumMovement.MONEY: "Money",
    AlbumMovement.US_AND_THEM: "Us and Them",
    AlbumMovement.ANY_COLOUR: "Any Colour",
    AlbumMovement.BRAIN_DAMAGE: "Brain Damage",
    AlbumMovement.ECLIPSE: "Eclipse",
}


class EffectEngineDarkSide:
    """Album mood wash + live chase of the most prominent instrument."""

    def __init__(self) -> None:
        self._hue = WHITE_BEAM[0]
        self._sat = WHITE_BEAM[1]
        self._val = 0.10
        self._spectrum_t = 0.0
        self._refract = 0.0
        self._heartbeat = 0.0
        self._bass_s = 0.0
        self._mid_s = 0.0
        self._high_s = 0.0
        self._sub_s = 0.0
        self._energy_s = 0.0
        self._flux_s = 0.0
        self._pulse = 0.0
        self._accent_until = 0.0
        self._clock_until = 0.0
        self._leslie = 0.0
        self._swell = 0.0
        self._lead_h = WHITE_BEAM[0]
        self._lead_s = 0.5
        self._lead_w = 0.0
        self._lead_name = "void"
        self._blend_s = 0.0  # smoothed mood↔lead mix
        self._mood_h = WHITE_BEAM[0]
        self._mood_s = WHITE_BEAM[1]
        self._voice_w = {"drums": 0.25, "guitar": 0.25, "synth": 0.25, "air": 0.25}
        self._movement = AlbumMovement.HEARTBEAT
        self._movement_since = time.perf_counter()
        self._energy_hist: deque[float] = deque(maxlen=48)
        self._high_hist: deque[float] = deque(maxlen=24)
        self._mid_hist: deque[float] = deque(maxlen=24)
        self.movement_label = MOVEMENT_LABEL[AlbumMovement.HEARTBEAT]

    @property
    def movement(self) -> AlbumMovement:
        return self._movement

    def render(self, intent: LightIntent, ctx: MusicContext) -> tuple[int, int, int]:
        now = time.perf_counter()

        # Extra-smooth band meters so colour doesn't jitter with every frame
        self._bass_s = 0.72 * self._bass_s + 0.28 * ctx.bass_energy
        self._mid_s = 0.74 * self._mid_s + 0.26 * ctx.mid_energy
        self._high_s = 0.68 * self._high_s + 0.32 * ctx.high_energy
        self._sub_s = 0.72 * self._sub_s + 0.28 * ctx.sub_energy
        self._energy_s = 0.80 * self._energy_s + 0.20 * ctx.energy
        self._flux_s = 0.75 * self._flux_s + 0.25 * ctx.flux
        self._energy_hist.append(self._energy_s)
        self._high_hist.append(self._high_s)
        self._mid_hist.append(self._mid_s)

        movement = self._classify(ctx, intent)

        self._leslie = (self._leslie + 0.0045 + intent.anim_speed * 0.003) % 1.0
        leslie_wobble = 0.5 + 0.5 * math.sin(self._leslie * math.tau)

        e_var = _variance(self._energy_hist)
        target_swell = _clamp01((self._energy_s - 0.22) / 0.55)
        if e_var > 0.012:
            target_swell = _clamp01(target_swell + e_var * 4.0)
        self._swell = 0.94 * self._swell + 0.06 * target_swell

        if (
            self._high_s > 0.38
            and self._high_s > self._mid_s * 1.35
            and self._flux_s > 0.12
        ):
            self._clock_until = now + 0.85

        # Lead voice: soft-mix drums / guitar / synth / air
        lead_h, lead_s, lead_w, lead_drive = self._follow_lead(ctx)
        lead_tag = {
            "drums": "drums",
            "guitar": "guitar",
            "synth": "synth",
            "air": "air",
        }.get(self._lead_name, "")
        self.movement_label = (
            f"{MOVEMENT_LABEL[movement]}-{lead_tag}" if lead_tag else MOVEMENT_LABEL[movement]
        )

        mood_h, mood_s, refract_t, crawl = self._look_for_movement(
            movement, intent, ctx, leslie_wobble, now
        )
        # Melt mood targets so movement changes don't hard-cut
        self._mood_h = _lerp_hue(self._mood_h, mood_h, 0.06)
        self._mood_s = 0.92 * self._mood_s + 0.08 * mood_s

        # Clear lead → ease toward it; never yank
        mood_bias = 0.55 if movement == AlbumMovement.HEARTBEAT else 0.28
        if movement == AlbumMovement.ANY_COLOUR:
            mood_bias = 0.18
        elif movement in (AlbumMovement.ON_THE_RUN, AlbumMovement.GREAT_GIG):
            mood_bias = 0.20
        blend_t = _clamp01(lead_w * (1.0 - mood_bias))
        self._blend_s = 0.88 * self._blend_s + 0.12 * blend_t
        target_h = _lerp_hue(self._mood_h, lead_h, self._blend_s)
        target_s = self._mood_s + (lead_s - self._mood_s) * self._blend_s

        ease_r = 0.035 if movement in (
            AlbumMovement.HEARTBEAT,
            AlbumMovement.BREATHE,
            AlbumMovement.US_AND_THEM,
        ) else 0.055
        if movement in (AlbumMovement.ON_THE_RUN, AlbumMovement.ECLIPSE) or intent.accent:
            ease_r = 0.10
        self._refract = self._refract + (refract_t - self._refract) * ease_r

        lead_crawl = 0.0
        if self._lead_name in ("synth", "guitar") and lead_w > 0.35:
            lead_crawl = 0.0008 + lead_w * 0.0025 + self._flux_s * 0.002
        self._spectrum_t = (self._spectrum_t + crawl + lead_crawl) % 1.0

        # Gentle spectrum glide on beats — no hard band hops
        if intent.color_step >= 1 and movement not in (
            AlbumMovement.HEARTBEAT,
            AlbumMovement.BREATHE,
        ):
            step = 1.0 / len(PRISM_SPECTRUM)
            jump = step * (0.55 if intent.color_step >= 2 else 0.28)
            if movement == AlbumMovement.MONEY:
                jump *= 1.10 if ctx.downbeat else 0.70
            elif movement == AlbumMovement.ANY_COLOUR:
                jump *= 1.05
            if self._lead_name == "drums" and ctx.beat:
                jump *= 1.05
            elif self._lead_name == "guitar":
                jump *= 0.35
            elif self._lead_name == "synth":
                jump *= 0.85
            self._spectrum_t = (self._spectrum_t + jump) % 1.0

        if intent.accent or (movement == AlbumMovement.ECLIPSE and ctx.drop_event):
            self._accent_until = now + 1.8
            self._pulse = max(self._pulse, 0.38)
            self._refract = max(self._refract, 0.75)

        if now < self._accent_until:
            # Long soft Eclipse bloom — no white hard-cut
            t = 1.0 - (self._accent_until - now) / 1.8
            ease = t * t * (3.0 - 2.0 * t)
            if ease < 0.35:
                target_h = _lerp_hue(target_h, WHITE_BEAM[0], 0.25 + ease * 0.55)
                target_s = target_s * (1.0 - ease * 0.7) + 0.06 * ease
            else:
                bloom = (ease - 0.35) / 0.65
                self._spectrum_t = (self._spectrum_t + bloom * 0.035) % 1.0
                ph, ps, _ = _sample_spectrum((self._spectrum_t + bloom * 0.35) % 1.0)
                target_h = _lerp_hue(target_h, ph, 0.20 + bloom * 0.45)
                target_s = target_s * (1.0 - bloom * 0.35) + ps * bloom * 0.55

        if now < self._clock_until and movement in (
            AlbumMovement.TIME,
            AlbumMovement.HEARTBEAT,
            AlbumMovement.ON_THE_RUN,
        ):
            # Soft clock shimmer — no strobe
            tick = 0.5 + 0.5 * math.sin(now * 14.0)
            target_h = _lerp_hue(target_h, 0.13, 0.18 + tick * 0.12)
            target_s = min(0.62, target_s * 0.82 + 0.08)
            self._pulse = max(self._pulse, 0.08 + tick * 0.10)

        # Slow colour melt — still tracks music, never snaps
        hue_follow = 0.045 + intent.anim_speed * 0.03
        if intent.color_step:
            hue_follow = 0.07
        hue_follow = max(hue_follow, 0.04 + lead_w * 0.10)
        if self._lead_name in ("synth", "guitar") and lead_w > 0.45:
            hue_follow = max(hue_follow, 0.08 + self._flux_s * 0.08)
        if self._lead_name == "drums" and (ctx.beat or ctx.kick_impact > 0.28):
            hue_follow = max(hue_follow, 0.10)
        if movement == AlbumMovement.ON_THE_RUN:
            hue_follow = max(hue_follow, 0.09)
        if now < self._accent_until:
            hue_follow = max(hue_follow, 0.08)
        self._hue = _lerp_hue(self._hue, target_h, hue_follow)
        self._sat = 0.92 * self._sat + 0.08 * _clamp01(target_s)

        brightness = self._brightness(
            movement, intent, ctx, leslie_wobble, now, lead_drive, lead_w
        )

        # High stickiness = lamp wash; beats only nudge
        follow = 0.42 if ctx.beat else 0.55
        if self._lead_name in ("guitar", "synth") and lead_w > 0.40:
            follow = 0.38
        if self._lead_name == "drums" and ctx.beat:
            follow = 0.32
        if movement == AlbumMovement.ON_THE_RUN:
            follow = 0.30
        elif movement == AlbumMovement.GREAT_GIG:
            follow = 0.45
        self._val = follow * self._val + (1.0 - follow) * brightness
        return _hsv_to_rgb(self._hue, self._sat, self._val)

    # ------------------------------------------------------------------ lead

    def _follow_lead(
        self, ctx: MusicContext
    ) -> tuple[float, float, float, float]:
        """Soft-mix drums / guitar / synth / air — morph between voices."""
        kick = max(ctx.kick_impact, ctx.onset_strength * 0.70)
        drums = (
            self._sub_s * 0.75
            + self._bass_s * 0.90
            + kick * 1.15
            + (0.32 if ctx.beat else 0.0)
            + (0.20 if ctx.downbeat else 0.0)
        )
        guitar = (
            self._mid_s * 1.25
            + max(0.0, self._mid_s - self._high_s) * 0.40
            - self._flux_s * 0.18
        )
        synth = (
            self._mid_s * 0.50
            + self._high_s * 0.60
            + self._flux_s * 1.05
            + max(0.0, ctx.energy_slope) * 0.45
        )
        air = (
            self._high_s * 1.15
            + ctx.onset_density * 0.40
            + self._flux_s * 0.18
        )

        raw = {
            "drums": max(0.05, drums),
            "guitar": max(0.05, guitar),
            "synth": max(0.05, synth),
            "air": max(0.05, air),
        }
        # Smooth voice weights so drums→guitar doesn't hard-cut colour
        total_raw = sum(raw.values())
        for name, val in raw.items():
            target_w = val / total_raw
            self._voice_w[name] = 0.82 * self._voice_w[name] + 0.18 * target_w
        # Renormalize
        tw = sum(self._voice_w.values()) + 1e-6
        for name in self._voice_w:
            self._voice_w[name] /= tw

        lead_name = max(self._voice_w, key=self._voice_w.get)
        lead_raw = raw[lead_name]
        second = sorted(raw.values(), reverse=True)[1]
        clarity = _clamp01((lead_raw - second) / (lead_raw + 1e-6))
        presence = _clamp01(lead_raw / 0.80)
        weight = _clamp01(0.25 + presence * 0.35 + clarity * 0.40)

        prism_h, _, _ = _sample_spectrum(self._spectrum_t)
        # Per-voice hues
        h_drums = _lerp_hue(0.00, 0.085, _clamp01(self._bass_s))
        h_drums = _lerp_hue(h_drums, 0.95, kick * 0.08)
        s_drums = 0.78 + kick * 0.10
        d_drums = self._bass_s * 0.42 + self._sub_s * 0.28 + kick * 0.45

        h_guitar = _lerp_hue(0.09, 0.38, _clamp01(self._mid_s * 0.95))
        h_guitar = _lerp_hue(h_guitar, prism_h, 0.30 + self._mid_s * 0.40)
        s_guitar = 0.68 + self._mid_s * 0.20
        d_guitar = self._mid_s * 0.78 + self._energy_s * 0.12

        h_synth = _lerp_hue(0.48, 0.82, _clamp01(self._high_s * 0.65 + self._flux_s))
        h_synth = _lerp_hue(h_synth, prism_h, 0.45 + self._flux_s * 0.40)
        s_synth = 0.74 + self._flux_s * 0.14
        d_synth = self._mid_s * 0.32 + self._high_s * 0.38 + self._flux_s * 0.45

        h_air = _lerp_hue(0.55, 0.80, _clamp01(self._high_s))
        s_air = 0.55 + self._high_s * 0.25
        d_air = self._high_s * 0.55 + ctx.onset_density * 0.28

        # Soft hue blend across all voices
        hues = {
            "drums": h_drums,
            "guitar": h_guitar,
            "synth": h_synth,
            "air": h_air,
        }
        sats = {
            "drums": s_drums,
            "guitar": s_guitar,
            "synth": s_synth,
            "air": s_air,
        }
        drives = {
            "drums": d_drums,
            "guitar": d_guitar,
            "synth": d_synth,
            "air": d_air,
        }
        # Weighted circular hue mix via successive lerps from dominant
        ordered = sorted(self._voice_w.items(), key=lambda kv: kv[1], reverse=True)
        h = hues[ordered[0][0]]
        s = sats[ordered[0][0]]
        drive = 0.0
        remaining = 1.0
        for name, w in ordered:
            if remaining <= 0:
                break
            mix = min(1.0, w / remaining) if remaining > 1e-6 else 0.0
            # First voice already set; subsequent voices ease in
            if name != ordered[0][0]:
                h = _lerp_hue(h, hues[name], mix * 0.85)
                s = s + (sats[name] - s) * mix * 0.85
            drive += drives[name] * w
            remaining -= w

        # Slow lead colour melt
        lead_follow = 0.08 + clarity * 0.10
        if lead_name == "drums" and (ctx.beat or kick > 0.32):
            lead_follow = 0.16
        elif lead_name in ("guitar", "synth") and presence > 0.45:
            lead_follow = max(lead_follow, 0.12)

        self._lead_h = _lerp_hue(self._lead_h, h, lead_follow)
        self._lead_s = 0.88 * self._lead_s + 0.12 * _clamp01(s)
        self._lead_w = 0.82 * self._lead_w + 0.18 * weight
        self._lead_name = lead_name
        return self._lead_h, self._lead_s, self._lead_w, _clamp01(drive)

    # ------------------------------------------------------------------ classify

    def _classify(self, ctx: MusicContext, intent: LightIntent) -> AlbumMovement:
        now = time.perf_counter()
        age = now - self._movement_since
        e = self._energy_s
        bass, mid, high = self._bass_s, self._mid_s, self._high_s
        bpm = ctx.bpm if ctx.bpm > 0 else 0.0
        dens = ctx.onset_density
        flux = self._flux_s
        e_var = _variance(self._energy_hist)
        state = ctx.state.value
        total = bass + mid + high + 1e-6
        w_bass, w_mid, w_high = bass / total, mid / total, high / total

        scores: dict[AlbumMovement, float] = {m: 0.0 for m in AlbumMovement}

        if e < 0.28 and w_bass >= w_high and dens < 0.18 and flux < 0.12:
            scores[AlbumMovement.HEARTBEAT] += 0.65 + max(0.0, 0.28 - e) * 2.0
            if w_bass >= 0.40:
                scores[AlbumMovement.HEARTBEAT] += 0.20

        if 0.10 < e < 0.50 and w_mid > 0.30 and flux < 0.14 and dens < 0.20:
            scores[AlbumMovement.BREATHE] += 0.50
            if bpm and bpm < 90:
                scores[AlbumMovement.BREATHE] += 0.35
            if state in ("chill", "groove", "breakdown"):
                scores[AlbumMovement.BREATHE] += 0.18
            if w_high < 0.28 and e_var < 0.012:
                scores[AlbumMovement.BREATHE] += 0.18

        if dens > 0.16 and flux > 0.08 and e > 0.25:
            run = dens * 1.0 + flux * 1.2 + w_high * 0.25
            scores[AlbumMovement.ON_THE_RUN] += 0.30 + run * 0.45
            if bpm >= 135:
                scores[AlbumMovement.ON_THE_RUN] += 0.45
            elif dens > 0.28:
                scores[AlbumMovement.ON_THE_RUN] += 0.22
            if ctx.tension > 0.30 or state == "buildup":
                scores[AlbumMovement.ON_THE_RUN] += 0.20

        if now < self._clock_until:
            scores[AlbumMovement.TIME] += 0.95
        if e > 0.35 and w_mid > 0.28 and w_bass > 0.22:
            if 95 <= bpm <= 145:
                scores[AlbumMovement.TIME] += 0.45
            if state in ("energetic", "peak", "groove"):
                scores[AlbumMovement.TIME] += 0.20
            if w_high < 0.35 and mid > 0.25:
                scores[AlbumMovement.TIME] += 0.15

        if w_mid > 0.40 and mid > 0.28 and e_var > 0.007 and self._swell > 0.28:
            scores[AlbumMovement.GREAT_GIG] += 0.30 + min(0.50, e_var * 20.0)
            if w_mid > w_bass and w_mid > w_high:
                scores[AlbumMovement.GREAT_GIG] += 0.22
            if self._swell > 0.50:
                scores[AlbumMovement.GREAT_GIG] += 0.22

        if e > 0.28 and w_bass > 0.30 and dens > 0.08:
            scores[AlbumMovement.MONEY] += 0.40
            if 100 <= bpm <= 150:
                scores[AlbumMovement.MONEY] += 0.42
            elif dens > 0.14 and w_bass >= w_mid:
                scores[AlbumMovement.MONEY] += 0.20
            if w_high > 0.18 and flux > 0.06:
                scores[AlbumMovement.MONEY] += 0.22
            if state in ("groove", "energetic", "peak"):
                scores[AlbumMovement.MONEY] += 0.18
            if w_bass > w_mid:
                scores[AlbumMovement.MONEY] += 0.15

        if e_var > 0.008 and w_mid > 0.28 and dens < 0.28:
            scores[AlbumMovement.US_AND_THEM] += 0.40 + min(0.40, e_var * 10.0)
            if bpm and bpm < 90:
                scores[AlbumMovement.US_AND_THEM] += 0.35
            if state in ("chill", "groove", "breakdown", "buildup"):
                scores[AlbumMovement.US_AND_THEM] += 0.15

        if e > 0.35 and w_mid > 0.28 and w_high > 0.18:
            scores[AlbumMovement.ANY_COLOUR] += 0.35
            if dens > 0.14:
                scores[AlbumMovement.ANY_COLOUR] += 0.18
            if 70 <= bpm <= 160 or bpm == 0:
                scores[AlbumMovement.ANY_COLOUR] += 0.12
            if w_bass < 0.38:
                scores[AlbumMovement.ANY_COLOUR] += 0.18
            if state in ("energetic", "peak", "groove"):
                scores[AlbumMovement.ANY_COLOUR] += 0.15

        if 0.20 < e < 0.55 and w_mid > 0.32 and w_bass < 0.40:
            scores[AlbumMovement.BRAIN_DAMAGE] += 0.35
            if bpm and 55 <= bpm <= 95:
                scores[AlbumMovement.BRAIN_DAMAGE] += 0.30
            if w_high > 0.20 and flux > 0.05:
                scores[AlbumMovement.BRAIN_DAMAGE] += 0.15
            if state in ("groove", "chill", "energetic"):
                scores[AlbumMovement.BRAIN_DAMAGE] += 0.10

        if intent.accent or ctx.drop_event:
            scores[AlbumMovement.ECLIPSE] += 0.85
        if state == "peak" and e > 0.60 and self._swell > 0.55:
            scores[AlbumMovement.ECLIPSE] += 0.45
        if state == "drop" and e > 0.55 and (ctx.drop_event or intent.accent):
            scores[AlbumMovement.ECLIPSE] += 0.35

        scores[self._movement] += 0.22 if age < 8.0 else 0.08

        min_dwell = 1.6
        if self._movement in (
            AlbumMovement.HEARTBEAT,
            AlbumMovement.BREATHE,
            AlbumMovement.US_AND_THEM,
        ):
            min_dwell = 2.0
        elif self._movement in (AlbumMovement.ON_THE_RUN, AlbumMovement.ECLIPSE):
            min_dwell = 1.2
        if age < min_dwell:
            return self._movement

        best = max(scores, key=scores.get)
        if best != self._movement and scores[best] >= scores[self._movement] + 0.10:
            self._movement = best
            self._movement_since = now
        return self._movement

    # ------------------------------------------------------------------ look

    def _look_for_movement(
        self,
        movement: AlbumMovement,
        intent: LightIntent,
        ctx: MusicContext,
        leslie: float,
        now: float,
    ) -> tuple[float, float, float, float]:
        prism_h, prism_s, _ = _sample_spectrum(self._spectrum_t)
        beam_h, beam_s, _ = WHITE_BEAM

        if movement == AlbumMovement.HEARTBEAT:
            foreshadow = 0.0
            if ctx.onset_strength > 0.35 or self._flux_s > 0.20:
                foreshadow = 0.35
                self._spectrum_t = (self._spectrum_t + 0.08) % 1.0
            h = _lerp_hue(beam_h, prism_h, foreshadow)
            s = beam_s + foreshadow * 0.45
            return h, s, 0.05 + foreshadow * 0.25, 0.0002

        if movement == AlbumMovement.BREATHE:
            warm = 0.08 + leslie * 0.04
            cool = 0.48
            h = _lerp_hue(warm, cool, 0.35 + leslie * 0.30)
            h = _lerp_hue(h, prism_h, 0.15 + self._mid_s * 0.25)
            s = 0.42 + self._mid_s * 0.25
            return h, s, 0.22 + self._energy_s * 0.25, 0.00045 + leslie * 0.0003

        if movement == AlbumMovement.ON_THE_RUN:
            scramble = 0.004 + self._flux_s * 0.012 + ctx.onset_density * 0.008
            h = _lerp_hue(0.78, prism_h, 0.40 + self._high_s * 0.35)
            if ctx.onset_strength > 0.3 or self._high_s > 0.45:
                h = _lerp_hue(h, 0.00, 0.35)
            s = 0.78 + self._flux_s * 0.18
            return h, s, 0.70 + ctx.tension * 0.25, scramble

        if movement == AlbumMovement.TIME:
            h = _lerp_hue(prism_h, 0.05, 0.15)
            if self._mid_s > 0.35:
                h = _lerp_hue(h, 0.12, 0.20)
            s = 0.72 + self._energy_s * 0.20
            crawl = 0.0012 + intent.anim_speed * 0.0025 + self._mid_s * 0.002
            return h, s, 0.55 + self._energy_s * 0.35, crawl

        if movement == AlbumMovement.GREAT_GIG:
            vocal = _clamp01(self._mid_s * 1.2)
            h = _lerp_hue(0.95, 0.78, 0.30 + vocal * 0.50 + self._swell * 0.20)
            if self._swell > 0.55:
                h = _lerp_hue(h, beam_h, (self._swell - 0.55) * 0.35)
            s = 0.55 + vocal * 0.38
            return h, s, 0.40 + vocal * 0.50, 0.0008 + vocal * 0.0035

        if movement == AlbumMovement.MONEY:
            h = _lerp_hue(0.13, 0.33, 0.35 + self._high_s * 0.40)
            h = _lerp_hue(h, 0.02, self._bass_s * 0.35)
            if ctx.onset_strength > 0.28 and self._high_s > self._bass_s * 0.7:
                h = _lerp_hue(h, 0.13, 0.55)
                s = 0.92
            else:
                s = 0.78 + self._bass_s * 0.15
            return h, s, 0.50 + self._energy_s * 0.35, 0.0010 + intent.anim_speed * 0.002

        if movement == AlbumMovement.US_AND_THEM:
            h = _lerp_hue(0.58, prism_h, self._swell)
            s = 0.35 + self._swell * 0.50
            return h, s, 0.15 + self._swell * 0.70, 0.0004 + self._swell * 0.0020

        if movement == AlbumMovement.ANY_COLOUR:
            crawl = 0.0022 + intent.anim_speed * 0.0035 + self._mid_s * 0.003
            return prism_h, min(0.96, prism_s), 0.88 + self._energy_s * 0.10, crawl

        if movement == AlbumMovement.BRAIN_DAMAGE:
            h = _lerp_hue(0.36, 0.78, 0.40 + self._mid_s * 0.35)
            if self._high_s > 0.28 and self._flux_s > 0.10:
                h = _lerp_hue(h, 0.88, 0.30)
            s = 0.55 + self._mid_s * 0.25
            return h, s, 0.35 + self._energy_s * 0.30, 0.0007

        h = _lerp_hue(prism_h, beam_h, 0.15 + self._swell * 0.25)
        s = 0.70 + self._energy_s * 0.20
        return h, s, 0.90, 0.0018 + self._energy_s * 0.003

    # ------------------------------------------------------------------ brightness

    def _brightness(
        self,
        movement: AlbumMovement,
        intent: LightIntent,
        ctx: MusicContext,
        leslie: float,
        now: float,
        lead_drive: float = 0.0,
        lead_w: float = 0.0,
    ) -> float:
        floor = intent.base_brightness * 0.26
        if movement in (AlbumMovement.HEARTBEAT, AlbumMovement.BREATHE):
            floor *= 0.50
        elif movement == AlbumMovement.US_AND_THEM:
            floor *= 0.40 + self._swell * 0.45

        # Base drive + strong pull from whichever voice is leading
        drive = (
            self._bass_s * 0.16
            + self._energy_s * 0.18
            + self._mid_s * 0.12
            + lead_drive * (0.35 + lead_w * 0.45)
        )

        hb = 0.0
        if ctx.bpm >= 40 or movement == AlbumMovement.HEARTBEAT:
            phase = ctx.beat_phase
            lub = max(0.0, math.cos(phase * math.tau)) ** 3.4
            dub = max(0.0, math.cos((phase - 0.18) * math.tau)) ** 4.2
            hb_amt = 0.12 + intent.beat_strength * 0.30
            if movement == AlbumMovement.HEARTBEAT:
                hb_amt *= 1.55
            elif movement in (AlbumMovement.BREATHE, AlbumMovement.US_AND_THEM):
                hb_amt *= 1.15
            elif movement in (AlbumMovement.MONEY, AlbumMovement.TIME):
                hb_amt *= 0.45
            # When drums lead, heartbeat / kick reads a bit harder — still soft
            if self._lead_name == "drums":
                hb_amt *= 1.10
            hb = (lub + dub * 0.55) * hb_amt * (0.40 + self._bass_s * 0.60)
        self._heartbeat = 0.82 * self._heartbeat + 0.18 * hb

        leslie_glow = 0.0
        if movement in (AlbumMovement.BREATHE, AlbumMovement.ECLIPSE):
            leslie_glow = (leslie - 0.5) * 0.06

        vocal_drive = 0.0
        if movement == AlbumMovement.GREAT_GIG or self._lead_name == "guitar":
            vocal_drive = self._mid_s * (0.36 if movement == AlbumMovement.GREAT_GIG else 0.18)
            vocal_drive += self._swell * 0.10

        run_twitch = 0.0
        if movement == AlbumMovement.ON_THE_RUN or self._lead_name == "synth":
            run_twitch = self._flux_s * 0.18 + ctx.onset_density * 0.10

        if ctx.beat:
            amount = intent.downbeat_strength if ctx.downbeat else intent.beat_strength
            amount *= intent.kick_scale
            if self._lead_name == "drums":
                amount *= 0.70
            elif self._lead_name in ("guitar", "synth"):
                amount *= 0.28
            elif movement == AlbumMovement.MONEY:
                amount *= 0.55
            elif movement == AlbumMovement.HEARTBEAT:
                amount *= 0.28
            elif movement == AlbumMovement.US_AND_THEM:
                amount *= 0.30 + self._swell * 0.30
            else:
                amount *= 0.40
            amount *= 0.45 + 0.55 * self._bass_s
            # Ease pulse up instead of slamming
            self._pulse = max(self._pulse, _clamp01(amount * 0.75))
            self._pulse = 0.55 * self._pulse + 0.45 * _clamp01(amount * 0.75)
        if ctx.kick_impact > 0.45 and self._lead_name == "drums":
            self._pulse = 0.70 * self._pulse + 0.30 * _clamp01(ctx.kick_impact * 0.45)
        self._pulse *= 0.88 if movement != AlbumMovement.ON_THE_RUN else 0.82

        brightness = _clamp01(
            floor
            + drive
            + self._heartbeat
            + self._pulse * 0.40
            + leslie_glow
            + vocal_drive
            + run_twitch
        )

        cap = {
            AlbumMovement.HEARTBEAT: 0.42,
            AlbumMovement.BREATHE: 0.52,
            AlbumMovement.US_AND_THEM: 0.62,
            AlbumMovement.BRAIN_DAMAGE: 0.58,
            AlbumMovement.ON_THE_RUN: 0.70,
            AlbumMovement.GREAT_GIG: 0.80,
            AlbumMovement.ECLIPSE: 0.82,
            AlbumMovement.MONEY: 0.72,
            AlbumMovement.TIME: 0.76,
            AlbumMovement.ANY_COLOUR: 0.76,
        }.get(movement, 0.68)
        if self._lead_name in ("guitar", "synth") and lead_w > 0.50:
            cap = min(0.85, cap + 0.06)
        brightness = min(brightness, cap)
        brightness = max(brightness, 0.015)

        if now < self._accent_until:
            t = (self._accent_until - now) / 1.8
            brightness = max(brightness, 0.36 + 0.22 * t)

        if intent.hush > 0:
            brightness = _clamp01(brightness * (1.0 - intent.hush * 0.72))

        return brightness


def _sample_spectrum(t: float) -> tuple[float, float, float]:
    bands = PRISM_SPECTRUM
    n = len(bands)
    x = (t % 1.0) * n
    i = int(x) % n
    j = (i + 1) % n
    f = x - int(x)
    h = _lerp_hue(bands[i][0], bands[j][0], f)
    s = bands[i][1] + (bands[j][1] - bands[i][1]) * f
    v = bands[i][2] + (bands[j][2] - bands[i][2]) * f
    return h, s, v


def _variance(values: deque[float]) -> float:
    if len(values) < 6:
        return 0.0
    m = sum(values) / len(values)
    return sum((v - m) ** 2 for v in values) / len(values)


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, _clamp01(s), _clamp01(v))
    return int(r * 255), int(g * 255), int(b * 255)


def _lerp_hue(a: float, b: float, t: float) -> float:
    d = (b - a + 0.5) % 1.0 - 0.5
    return (a + d * _clamp01(t)) % 1.0


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
