from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys

from rgb_lights.ble.protocols import DEVICE_ADDRESS, WRITE_CHARACTERISTIC

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


@dataclass
class DeviceConfig:
    address: str = DEVICE_ADDRESS
    characteristic: str = WRITE_CHARACTERISTIC


@dataclass
class AudioConfig:
    source: str = "system"  # system | mic
    sample_rate: int = 44100
    chunk_size: int = 2048


@dataclass
class ReactiveConfig:
    update_hz: float = 12.0
    smoothing: float = 0.4
    gain: float = 1.6
    mode: str = "party"  # party | calm | darkside | festival | spectrum | bass_pulse | energy
    delay_ms: float = 200.0
    # Fire lights this many ms early to cancel analysis chunk + BLE path lag.
    lead_ms: float = 50.0
    # Global dorm comfort ceiling (0.15–1.0). Party pipeline uses this.
    intensity: float = 0.72


@dataclass
class MovieConfig:
    """Screen-sync / Ambilight-style movie mode (Hue Sync inspired)."""

    update_hz: float = 18.0
    # subtle | cinema | immersive | maximum  (moderate/high/extreme kept as aliases)
    intensity: str = "cinema"
    monitor: int = 1
    # Override intensity preset saturation gain when set (>0)
    saturation: float = 0.0
    # Soft floor — cinema mode prefers true dark (0). Raise if you want a night glow.
    min_brightness: float = 0.0
    # Override temporal smoothing when set in (0, 1] — stretches attack/release
    smoothing: float = 0.0
    # Outer frame weight band (room bleed). ~0.30 = outer 30%
    edge_depth: float = 0.32
    blackbar: bool = True
    # letterbox = top/bottom only (safer on PC); full = also crop sides
    blackbar_mode: str = "letterbox"
    # Per-channel balance from phone color calibrator (1.0 = neutral)
    r_gain: float = 1.0
    g_gain: float = 1.0
    b_gain: float = 1.0
    # 3x3 row-major color correction matrix (identity = no change)
    color_matrix: list[float] = field(
        default_factory=lambda: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    )
    # Highlight vs general illumination mix (explosions / sun)
    highlight_mix: float = 0.25
    # Scene luminance below this → treat as black (fade strip off)
    black_luma: float = 0.028


@dataclass
class AppConfig:
    device: DeviceConfig = field(default_factory=DeviceConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    reactive: ReactiveConfig = field(default_factory=ReactiveConfig)
    movie: MovieConfig = field(default_factory=MovieConfig)


def default_config_path() -> Path:
    return Path.cwd() / "config.toml"


def load_config(path: Path | None = None) -> AppConfig:
    path = path or default_config_path()
    if not path.exists():
        return AppConfig()

    with path.open("rb") as f:
        raw = tomllib.load(f)

    device_raw = {
        k: v
        for k, v in raw.get("device", {}).items()
        if k in DeviceConfig.__annotations__ and v not in ("", None)
    }
    device = DeviceConfig(**device_raw)
    audio = AudioConfig(**{k: v for k, v in raw.get("audio", {}).items() if k in AudioConfig.__annotations__})
    reactive = ReactiveConfig(
        **{k: v for k, v in raw.get("reactive", {}).items() if k in ReactiveConfig.__annotations__}
    )
    movie_raw = {
        k: v for k, v in raw.get("movie", {}).items() if k in MovieConfig.__annotations__
    }
    if "color_matrix" in movie_raw and not isinstance(movie_raw["color_matrix"], list):
        movie_raw.pop("color_matrix", None)
    movie = MovieConfig(**movie_raw)
    return AppConfig(device=device, audio=audio, reactive=reactive, movie=movie)
