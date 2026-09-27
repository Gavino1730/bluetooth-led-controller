from __future__ import annotations

from dataclasses import dataclass
import queue
import sys
import threading
import numpy as np


@dataclass
class AudioFrame:
    samples: np.ndarray  # float32 mono, -1..1
    sample_rate: int


class AudioCapture:
    """Capture system audio (preferred) or microphone into mono float frames."""

    def __init__(
        self,
        source: str = "system",
        sample_rate: int = 44100,
        chunk_size: int = 2048,
    ) -> None:
        self.source = source.lower().strip()
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=8)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._backend_name = ""
        self._device_label = ""

    @property
    def backend_name(self) -> str:
        return self._backend_name

    @property
    def device_label(self) -> str:
        return self._device_label

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        if sys.platform == "win32" and self.source == "system":
            target = self._run_wasapi_loopback
        else:
            target = self._run_sounddevice
        self._thread = threading.Thread(target=target, name="audio-capture", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def read(self, timeout: float = 1.0) -> AudioFrame | None:
        try:
            samples = self._queue.get(timeout=timeout)
        except queue.Empty:
            return None
        return AudioFrame(samples=samples, sample_rate=self.sample_rate)

    def _enqueue(self, mono: np.ndarray) -> None:
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self._queue.put_nowait(mono.astype(np.float32, copy=False))
        except queue.Full:
            pass

    def _run_wasapi_loopback(self) -> None:
        try:
            import pyaudiowpatch as pyaudio
        except ImportError as exc:
            raise RuntimeError(
                "System audio capture on Windows needs PyAudioWPatch. "
                "Install with: pip install PyAudioWPatch"
            ) from exc

        self._backend_name = "pyaudiowpatch/WASAPI"
        pa = pyaudio.PyAudio()
        stream = None
        try:
            wasapi_info = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
            speakers = pa.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
            loopback = speakers
            if not speakers.get("isLoopbackDevice"):
                for candidate in pa.get_loopback_device_info_generator():
                    if speakers["name"] in candidate["name"]:
                        loopback = candidate
                        break
                else:
                    raise RuntimeError(
                        "Could not find a WASAPI loopback device for your default speakers. "
                        "Run: python -m pyaudiowpatch"
                    )

            channels = max(1, int(loopback["maxInputChannels"]))
            rate = int(loopback["defaultSampleRate"])
            self.sample_rate = rate
            self._device_label = f"{loopback['name']} (index {loopback['index']})"

            def callback(in_data, frame_count, time_info, status):  # noqa: ANN001
                if self._stop.is_set():
                    return (None, pyaudio.paComplete)
                audio = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
                if channels > 1:
                    audio = audio.reshape(-1, channels).mean(axis=1)
                self._enqueue(audio)
                return (in_data, pyaudio.paContinue)

            stream = pa.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=rate,
                frames_per_buffer=self.chunk_size,
                input=True,
                input_device_index=int(loopback["index"]),
                stream_callback=callback,
            )
            stream.start_stream()
            while not self._stop.is_set() and stream.is_active():
                self._stop.wait(0.1)
        finally:
            if stream is not None:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
            pa.terminate()

    def _run_sounddevice(self) -> None:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError(
                "Microphone / non-Windows capture needs sounddevice. "
                "Install with: pip install sounddevice"
            ) from exc

        self._backend_name = "sounddevice"
        device = None
        if self.source == "system":
            # Best-effort: some Linux setups expose a monitor device as default input.
            self._device_label = "default input (set OS default to a monitor/loopback source)"
        else:
            self._device_label = "default microphone"

        def callback(indata, frames, time_info, status):  # noqa: ANN001
            if status:
                pass
            mono = indata.mean(axis=1) if indata.ndim > 1 else indata.reshape(-1)
            self._enqueue(np.asarray(mono, dtype=np.float32))

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=2,
            dtype="float32",
            blocksize=self.chunk_size,
            device=device,
            callback=callback,
        ):
            while not self._stop.wait(0.1):
                pass


def list_audio_devices() -> None:
    if sys.platform == "win32":
        try:
            import pyaudiowpatch as pyaudio
        except ImportError:
            print("PyAudioWPatch not installed.")
            return
        pa = pyaudio.PyAudio()
        print("WASAPI loopback devices:")
        any_found = False
        for device in pa.get_loopback_device_info_generator():
            any_found = True
            print(f"  [{device['index']}] {device['name']}  {int(device['defaultSampleRate'])} Hz")
        if not any_found:
            print("  (none)")
        pa.terminate()
        return

    try:
        import sounddevice as sd
    except ImportError:
        print("sounddevice not installed.")
        return
    print(sd.query_devices())
