from __future__ import annotations

"""Measure Bluetooth speaker lag relative to WASAPI loopback.

Loopback hears audio when Windows mixes it. A Bluetooth speaker plays it later.
We play short clicks, detect each click in loopback and on the microphone
(hearing the speaker), then take the median time difference.
"""

from collections import deque
from dataclasses import dataclass
import time
from pathlib import Path
import threading
import re

import numpy as np


@dataclass
class DelayEstimate:
    delay_ms: float
    samples: list[float]
    loopback_device: str
    mic_device: str


def measure_speaker_delay(
    clicks: int = 6,
    click_gap_s: float = 0.85,
    threshold: float = 0.12,
) -> DelayEstimate:
    """Return estimated loopback→speaker delay in milliseconds."""
    if clicks < 3:
        raise ValueError("Need at least 3 clicks for a stable median")

    try:
        import pyaudiowpatch as pyaudio
    except ImportError as exc:
        raise RuntimeError("PyAudioWPatch is required for delay calibration on Windows") from exc

    pa = pyaudio.PyAudio()
    loop_idx, loop_rate, loop_ch, loop_name = _find_loopback(pa, pyaudio)
    mic_idx, mic_rate, mic_ch, mic_name = _find_microphone(pa, pyaudio)

    loop_events: deque[float] = deque(maxlen=32)
    mic_events: deque[float] = deque(maxlen=32)
    stop = threading.Event()
    lock = threading.Lock()

    def make_callback(store: deque[float], channels: int, cooldown: list[float]):
        def callback(in_data, frame_count, time_info, status):  # noqa: ANN001
            if stop.is_set():
                return (None, pyaudio.paComplete)
            audio = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
            if channels > 1:
                audio = audio.reshape(-1, channels).mean(axis=1)
            peak = float(np.max(np.abs(audio))) if audio.size else 0.0
            now = time.perf_counter()
            if peak >= threshold and (now - cooldown[0]) > 0.25:
                cooldown[0] = now
                with lock:
                    store.append(now)
            return (in_data, pyaudio.paContinue)

        return callback

    loop_cd = [0.0]
    mic_cd = [0.0]
    loop_stream = pa.open(
        format=pyaudio.paInt16,
        channels=loop_ch,
        rate=loop_rate,
        frames_per_buffer=512,
        input=True,
        input_device_index=loop_idx,
        stream_callback=make_callback(loop_events, loop_ch, loop_cd),
    )
    mic_stream = pa.open(
        format=pyaudio.paInt16,
        channels=mic_ch,
        rate=mic_rate,
        frames_per_buffer=512,
        input=True,
        input_device_index=mic_idx,
        stream_callback=make_callback(mic_events, mic_ch, mic_cd),
    )
    loop_stream.start_stream()
    mic_stream.start_stream()

    # Warm up streams
    time.sleep(0.4)
    with lock:
        loop_events.clear()
        mic_events.clear()

    print(f"Loopback: {loop_name}")
    print(f"Mic:      {mic_name}")
    print(f"Playing {clicks} calibration clicks through your Bluetooth speaker...")
    print("Keep the room fairly quiet; put the mic where you hear the speaker.\n")

    try:
        for i in range(clicks):
            print(f"  click {i + 1}/{clicks}")
            _play_click(pa, pyaudio, sample_rate=44100)
            time.sleep(click_gap_s)
        # Catch the last mic arrival
        time.sleep(0.6)
    finally:
        stop.set()
        time.sleep(0.15)
        loop_stream.stop_stream()
        loop_stream.close()
        mic_stream.stop_stream()
        mic_stream.close()
        pa.terminate()

    with lock:
        loop_times = list(loop_events)
        mic_times = list(mic_events)

    pairs = _pair_events(loop_times, mic_times, max_lag_s=0.8)
    if len(pairs) < 2:
        raise RuntimeError(
            "Could not pair loopback and mic clicks. "
            "Check that Spotify/other apps are quiet, the mic can hear the speaker, "
            "and Windows default output is the Bluetooth speaker."
        )

    lags_ms = [(mic - loop) * 1000.0 for loop, mic in pairs]
    # Discard negative (mic before loopback) — usually false triggers
    lags_ms = [x for x in lags_ms if x > 20.0]
    if len(lags_ms) < 2:
        raw = [round((mic - loop) * 1000, 1) for loop, mic in pairs]
        raise RuntimeError(
            f"Measured lags looked invalid: {raw}. "
            "Try again closer to the speaker or raise Windows mic volume."
        )

    delay = float(np.median(lags_ms))
    return DelayEstimate(
        delay_ms=delay,
        samples=lags_ms,
        loopback_device=loop_name,
        mic_device=mic_name,
    )


def write_delay_to_config(delay_ms: float, path: Path | None = None) -> Path:
    path = path or Path.cwd() / "config.toml"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    rounded = int(round(delay_ms))
    if "delay_ms" in text:
        text = re.sub(r"(?m)^(delay_ms\s*=\s*)-?\d+(\.\d+)?", rf"\g<1>{rounded}", text)
    else:
        if "[reactive]" not in text:
            text += "\n[reactive]\n"
        text += f"delay_ms = {rounded}\n"
    path.write_text(text, encoding="utf-8")
    return path


def _find_loopback(pa, pyaudio):  # noqa: ANN001
    wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
    speakers = pa.get_device_info_by_index(wasapi["defaultOutputDevice"])
    loopback = speakers
    if not speakers.get("isLoopbackDevice"):
        for candidate in pa.get_loopback_device_info_generator():
            if speakers["name"] in candidate["name"]:
                loopback = candidate
                break
        else:
            raise RuntimeError("No WASAPI loopback device found for default speakers")
    return (
        int(loopback["index"]),
        int(loopback["defaultSampleRate"]),
        max(1, int(loopback["maxInputChannels"])),
        str(loopback["name"]),
    )


def _find_microphone(pa, pyaudio):  # noqa: ANN001
    # Prefer default input if it is not a loopback device.
    try:
        info = pa.get_default_input_device_info()
        if info and not info.get("isLoopbackDevice"):
            return (
                int(info["index"]),
                int(info["defaultSampleRate"]),
                max(1, int(info["maxInputChannels"])),
                str(info["name"]),
            )
    except Exception:
        pass
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info.get("maxInputChannels", 0) > 0 and not info.get("isLoopbackDevice"):
            return (
                int(info["index"]),
                int(info["defaultSampleRate"]),
                max(1, int(info["maxInputChannels"])),
                str(info["name"]),
            )
    raise RuntimeError("No microphone found for acoustic calibration")


def _play_click(pa, pyaudio, sample_rate: int = 44100) -> None:  # noqa: ANN001
    """Play a short band-limited click on the default output device."""
    duration = 0.035
    n = int(sample_rate * duration)
    t = np.arange(n, dtype=np.float32) / sample_rate
    # Decaying 1 kHz burst — easy to detect on both loopback and mic
    click = 0.9 * np.sin(2 * np.pi * 1000 * t) * np.exp(-t * 80)
    # Soft pre-silence so streams are settled
    audio = np.concatenate([np.zeros(int(0.02 * sample_rate), dtype=np.float32), click])
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()

    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        output=True,
        frames_per_buffer=512,
    )
    try:
        stream.write(pcm)
    finally:
        stream.stop_stream()
        stream.close()


def _pair_events(
    loop_times: list[float],
    mic_times: list[float],
    max_lag_s: float = 0.8,
) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    used_mic: set[int] = set()
    for lt in loop_times:
        best_i = None
        best_lag = None
        for i, mt in enumerate(mic_times):
            if i in used_mic:
                continue
            lag = mt - lt
            if 0.02 <= lag <= max_lag_s and (best_lag is None or lag < best_lag):
                best_lag = lag
                best_i = i
        if best_i is not None:
            used_mic.add(best_i)
            pairs.append((lt, mic_times[best_i]))
    return pairs
