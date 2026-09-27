from __future__ import annotations

from dataclasses import dataclass

from rgb_lights.audio.analyzer import AudioAnalyzer, AudioFeatures
from rgb_lights.lighting.calm import CalmEngine, CalmFrame, CalmMood
from rgb_lights.lighting.director import LightIntent, LightingDirector
from rgb_lights.lighting.effects import EffectEngine
from rgb_lights.lighting.effects_darkside import EffectEngineDarkSide
from rgb_lights.lighting.effects_v2 import EffectEngineV2
from rgb_lights.music.state import MusicContext, MusicStateTracker


@dataclass
class FrameResult:
    rgb: tuple[int, int, int]
    features: AudioFeatures
    context: MusicContext | None = None
    intent: LightIntent | None = None
    mood: CalmMood | None = None


class LightingPipeline:
    """Audio → musical state → director → effect → RGB.

    variant:
      v1        — palette party mode (default)
      calm      — mood washes (warmth/presence/space) — no beat grid
      darkside  — Pink Floyd DSOTM prism / heartbeat metaphor
      party2    — timbre→color association + structure brightness
      festival2 — same association, snappier / bigger swings
    """

    def __init__(
        self,
        intensity: float = 0.85,
        gain: float = 1.8,
        smoothing: float = 0.35,
        variant: str = "v1",
    ) -> None:
        self.variant = variant
        self._calm: CalmEngine | None = None
        self.analyzer: AudioAnalyzer | None = None
        self.state: MusicStateTracker | None = None
        self.director: LightingDirector | None = None
        self.effects = None

        if variant == "calm":
            self._calm = CalmEngine(
                intensity=intensity,
                gain=gain,
                smoothing=max(0.5, smoothing),
            )
            return

        self.analyzer = AudioAnalyzer(gain=gain, smoothing=smoothing)
        self.state = MusicStateTracker()
        director_style = "darkside" if variant == "darkside" else "party"
        self.director = LightingDirector(intensity=intensity, style=director_style)
        if variant in ("party2", "festival2"):
            self.effects = EffectEngineV2(style=variant)
        elif variant == "darkside":
            self.effects = EffectEngineDarkSide()
        else:
            self.effects = EffectEngine(style="party")

    def process(self, samples, sample_rate: int) -> FrameResult:
        if self._calm is not None:
            frame: CalmFrame = self._calm.process(samples, sample_rate)
            return FrameResult(rgb=frame.rgb, features=frame.features, mood=frame.mood)

        assert self.analyzer is not None and self.state is not None
        assert self.director is not None and self.effects is not None
        features = self.analyzer.analyze(samples, sample_rate)
        context = self.state.update(features, self.analyzer)
        intent = self.director.update(context)
        rgb = self.effects.render(intent, context)
        # Surface album movement on the status line for darkside
        if self.variant == "darkside" and hasattr(self.effects, "movement_label"):
            intent = LightIntent(
                palette=intent.palette,
                base_brightness=intent.base_brightness,
                beat_strength=intent.beat_strength,
                downbeat_strength=intent.downbeat_strength,
                anim_speed=intent.anim_speed,
                hue_mix=intent.hue_mix,
                color_step=intent.color_step,
                accent=intent.accent,
                accent_warm=intent.accent_warm,
                label=self.effects.movement_label,
                kick_scale=intent.kick_scale,
                tension_drive=intent.tension_drive,
                hush=intent.hush,
            )
        return FrameResult(rgb=rgb, features=features, context=context, intent=intent)
