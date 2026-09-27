from __future__ import annotations

"""Interactive visual delay calibration (no microphone).

Plays a tone on the default Windows output, then flashes the LED strip
(and an on-screen panel) after delay_ms. Adjust until the flash lines up
with the beep you hear from the Bluetooth speaker.

Uses a single-thread asyncio + Tk pump so Bleak/WinRT stays on the main
thread (background-thread Bleak connects often hang on Windows).
"""

import asyncio
import queue
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rgb_lights.ble import BleLightController
from rgb_lights.calibrate import write_delay_to_config
from rgb_lights.config import AppConfig


FLASH_RGB = (255, 220, 80)
IDLE_RGB = (8, 8, 12)
SAMPLE_RATE = 44100
BEEP_GAP_S = 1.05
TK_PUMP_S = 0.02


@dataclass
class CalibrateResult:
    delay_ms: float
    saved: bool
    path: Path | None


def _make_beep_pcm(sample_rate: int = SAMPLE_RATE) -> bytes:
    duration = 0.055
    n = int(sample_rate * duration)
    t = np.arange(n, dtype=np.float32) / sample_rate
    tone = 0.85 * np.sin(2 * np.pi * 880.0 * t) * np.exp(-t * 28.0)
    pad = np.zeros(int(0.015 * sample_rate), dtype=np.float32)
    audio = np.concatenate([pad, tone])
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


class _TonePlayer:
    def __init__(self) -> None:
        try:
            import pyaudiowpatch as pyaudio
        except ImportError as exc:
            raise RuntimeError(
                "PyAudioWPatch is required for visual calibration on Windows"
            ) from exc
        self._pyaudio_mod = pyaudio
        self._pa = pyaudio.PyAudio()
        self._pcm = _make_beep_pcm()
        self._lock = threading.Lock()

    def play(self) -> None:
        def _worker() -> None:
            with self._lock:
                stream = self._pa.open(
                    format=self._pyaudio_mod.paInt16,
                    channels=1,
                    rate=SAMPLE_RATE,
                    output=True,
                    frames_per_buffer=512,
                )
                try:
                    stream.write(self._pcm)
                finally:
                    stream.stop_stream()
                    stream.close()

        threading.Thread(target=_worker, name="tone-play", daemon=True).start()

    def close(self) -> None:
        with self._lock:
            self._pa.terminate()


class VisualCalibrator:
    def __init__(
        self,
        config: AppConfig,
        *,
        address: str | None = None,
        preview: bool = False,
        start_delay_ms: float | None = None,
        config_path: Path | None = None,
    ) -> None:
        self.config = config
        self.preview = preview
        self.config_path = config_path or Path.cwd() / "config.toml"
        self.delay_ms = float(
            start_delay_ms if start_delay_ms is not None else config.reactive.delay_ms
        )
        self.delay_ms = float(np.clip(self.delay_ms, 0.0, 600.0))
        self._address = address or config.device.address
        self._controller: BleLightController | None = None
        self._lights_ready = False
        self._player: _TonePlayer | None = None
        self._running = False
        self._closing = False
        self._flash_jobs: queue.Queue[float] = queue.Queue()
        self._result = CalibrateResult(delay_ms=self.delay_ms, saved=False, path=None)
        self._loop = asyncio.new_event_loop()
        self._build_ui()

    def _build_ui(self) -> None:
        self.root = tk.Tk()
        self.root.title("RGB Lights — delay calibrate")
        self.root.geometry("720x520")
        self.root.minsize(560, 420)
        self.root.configure(bg="#0d0f14")

        self.flash = tk.Frame(self.root, bg="#12151c", height=260)
        self.flash.pack(fill=tk.BOTH, expand=True, padx=18, pady=(18, 8))
        self.flash.pack_propagate(False)

        self.flash_label = tk.Label(
            self.flash,
            text="FLASH",
            font=("Segoe UI Semibold", 42),
            fg="#3a4254",
            bg="#12151c",
        )
        self.flash_label.place(relx=0.5, rely=0.5, anchor="center")

        panel = tk.Frame(self.root, bg="#0d0f14")
        panel.pack(fill=tk.X, padx=18, pady=(0, 16))

        self.hint = tk.Label(
            panel,
            text=(
                "A beep plays on your speakers. Lights (and this panel) flash after the delay.\n"
                "Raise delay if the flash is early · lower if the flash is late · match beep to flash."
            ),
            font=("Segoe UI", 11),
            fg="#c6cddb",
            bg="#0d0f14",
            justify=tk.LEFT,
        )
        self.hint.pack(anchor="w", pady=(0, 10))

        self.delay_var = tk.DoubleVar(value=self.delay_ms)
        self.delay_label = tk.Label(
            panel,
            text=self._delay_text(),
            font=("Consolas", 20),
            fg="#f2f5fb",
            bg="#0d0f14",
        )
        self.delay_label.pack(anchor="w")

        self.slider = tk.Scale(
            panel,
            from_=0,
            to=500,
            orient=tk.HORIZONTAL,
            resolution=1,
            showvalue=False,
            variable=self.delay_var,
            command=self._on_slider,
            length=640,
            bg="#0d0f14",
            fg="#f2f5fb",
            troughcolor="#1c2230",
            highlightthickness=0,
            activebackground="#f0c24b",
        )
        self.slider.pack(fill=tk.X, pady=(4, 10))

        keys = tk.Label(
            panel,
            text="← / →  ±10 ms    Shift+← / →  ±1 ms    Space  beep now    S  save    Esc  quit",
            font=("Segoe UI", 10),
            fg="#8b93a7",
            bg="#0d0f14",
        )
        keys.pack(anchor="w", pady=(0, 10))

        row = tk.Frame(panel, bg="#0d0f14")
        row.pack(fill=tk.X)

        self.status = tk.Label(
            row,
            text="Starting…",
            font=("Segoe UI", 10),
            fg="#8b93a7",
            bg="#0d0f14",
            wraplength=420,
            justify=tk.LEFT,
        )
        self.status.pack(side=tk.LEFT, fill=tk.X, expand=True)

        btns = tk.Frame(row, bg="#0d0f14")
        btns.pack(side=tk.RIGHT)
        tk.Button(btns, text="Retry BLE", command=self._retry_ble, width=10).pack(side=tk.LEFT, padx=4)
        tk.Button(btns, text="Beep now", command=self._beep_once, width=10).pack(side=tk.LEFT, padx=4)
        tk.Button(btns, text="Save", command=self._save, width=10).pack(side=tk.LEFT, padx=4)
        tk.Button(btns, text="Quit", command=self._quit, width=10).pack(side=tk.LEFT, padx=4)

        self.root.bind("<Left>", lambda _e: self._nudge(-10))
        self.root.bind("<Right>", lambda _e: self._nudge(10))
        self.root.bind("<Shift-Left>", lambda _e: self._nudge(-1))
        self.root.bind("<Shift-Right>", lambda _e: self._nudge(1))
        self.root.bind("<space>", lambda _e: self._beep_once())
        self.root.bind("<s>", lambda _e: self._save())
        self.root.bind("<S>", lambda _e: self._save())
        self.root.bind("<Escape>", lambda _e: self._quit())
        self.root.protocol("WM_DELETE_WINDOW", self._quit)

    def run(self) -> CalibrateResult:
        asyncio.set_event_loop(self._loop)
        self._player = _TonePlayer()
        self._running = True
        self._loop.create_task(self._startup())
        self._loop.call_soon(self._tk_pump)
        self._loop.call_later(0.25, self._schedule_cycle)
        self._loop.call_later(0.05, self._poll_flashes)
        try:
            self._loop.run_forever()
        finally:
            self._cleanup_sync()
        return self._result

    def _tk_pump(self) -> None:
        if self._closing:
            return
        try:
            self.root.update()
        except tk.TclError:
            self._quit()
            return
        self._loop.call_later(TK_PUMP_S, self._tk_pump)

    async def _startup(self) -> None:
        if self.preview:
            self._set_status("Preview mode — screen flash only (no BLE)")
            print("Preview mode: screen flash only")
            return
        await self._connect_ble()

    async def _connect_ble(self) -> None:
        self._lights_ready = False
        self._set_status(f"Connecting to {self._address}…")
        print(f"Connecting to lights at {self._address} …")
        print("  (Close the phone app / any other rgb_lights session first)")
        controller = BleLightController(
            address=self._address,
            characteristic=self.config.device.characteristic or None,
        )
        try:
            await asyncio.wait_for(controller.connect(timeout=8.0), timeout=12.0)
            await controller.power_on()
            await controller.set_brightness(100)
            await controller.set_rgb(*IDLE_RGB, force=True)
        except Exception as exc:  # noqa: BLE001
            try:
                await controller.disconnect()
            except Exception:
                pass
            self._controller = None
            self._lights_ready = False
            name = type(exc).__name__
            if "NotFound" in name or "not found" in str(exc).lower():
                tip = (
                    f"Lights not found ({self._address}). "
                    "Power-cycle the strip, force-close the phone app, "
                    "then Retry BLE — screen flashes still work."
                )
            else:
                tip = f"BLE failed: {exc} — screen-only; Retry BLE when ready"
            self._set_status(tip)
            print(tip)
            return

        self._controller = controller
        self._lights_ready = True
        self._set_status("Connected — match flash to beep")
        print("Lights connected.")

    def _retry_ble(self) -> None:
        if self.preview or self._closing:
            return
        self._loop.create_task(self._reconnect())

    async def _reconnect(self) -> None:
        if self._controller is not None:
            try:
                await self._controller.disconnect()
            except Exception:
                pass
            self._controller = None
        self._lights_ready = False
        await self._connect_ble()

    def _delay_text(self) -> str:
        return f"delay_ms = {int(round(self.delay_ms))}"

    def _set_status(self, text: str) -> None:
        try:
            self.status.configure(text=text)
        except tk.TclError:
            pass

    def _on_slider(self, _value: str) -> None:
        self.delay_ms = float(self.delay_var.get())
        self.delay_label.configure(text=self._delay_text())
        self._result.delay_ms = self.delay_ms

    def _nudge(self, delta: float) -> None:
        self.delay_ms = float(np.clip(self.delay_ms + delta, 0.0, 600.0))
        self.delay_var.set(self.delay_ms)
        self.delay_label.configure(text=self._delay_text())
        self._result.delay_ms = self.delay_ms

    def _schedule_cycle(self) -> None:
        if not self._running or self._closing:
            return
        self._beep_once()
        self._loop.call_later(BEEP_GAP_S, self._schedule_cycle)

    def _beep_once(self) -> None:
        if self._player is None or self._closing:
            return
        self._player.play()
        ready_at = time.perf_counter() + (self.delay_ms / 1000.0)
        self._flash_jobs.put(ready_at)

    def _poll_flashes(self) -> None:
        if not self._running or self._closing:
            return
        now = time.perf_counter()
        due = False
        pending: list[float] = []
        while True:
            try:
                t = self._flash_jobs.get_nowait()
            except queue.Empty:
                break
            if t <= now:
                due = True
            else:
                pending.append(t)
        for t in pending:
            self._flash_jobs.put(t)
        if due:
            self._loop.create_task(self._do_flash())
        self._loop.call_later(0.008, self._poll_flashes)

    async def _do_flash(self) -> None:
        try:
            self.flash.configure(bg="#f5f0de")
            self.flash_label.configure(bg="#f5f0de", fg="#1a1408", text="NOW")
        except tk.TclError:
            return
        if self._lights_ready and self._controller is not None:
            try:
                await self._controller.set_rgb(*FLASH_RGB, force=True)
            except Exception as exc:  # noqa: BLE001
                self._lights_ready = False
                self._set_status(f"BLE write failed: {exc}")
        await asyncio.sleep(0.09)
        try:
            self.flash.configure(bg="#12151c")
            self.flash_label.configure(bg="#12151c", fg="#3a4254", text="FLASH")
        except tk.TclError:
            return
        if self._lights_ready and self._controller is not None:
            try:
                await self._controller.set_rgb(*IDLE_RGB, force=True)
            except Exception:
                pass

    def _save(self) -> None:
        path = write_delay_to_config(self.delay_ms, self.config_path)
        self._result = CalibrateResult(delay_ms=self.delay_ms, saved=True, path=path)
        self._set_status(f"Saved delay_ms = {int(round(self.delay_ms))} → {path}")
        print(f"Saved delay_ms = {int(round(self.delay_ms))} to {path}")

    def _quit(self) -> None:
        if self._closing:
            return
        self._closing = True
        self._running = False
        self._result.delay_ms = self.delay_ms
        self._loop.create_task(self._shutdown())

    async def _shutdown(self) -> None:
        if self._controller is not None:
            try:
                await self._controller.set_rgb(0, 0, 0, force=True)
                await self._controller.power_off()
            except Exception:
                pass
            try:
                await self._controller.disconnect()
            except Exception:
                pass
            self._controller = None
        self._loop.call_soon(self._loop.stop)

    def _cleanup_sync(self) -> None:
        if self._player is not None:
            self._player.close()
            self._player = None
        try:
            self.root.destroy()
        except tk.TclError:
            pass


def run_visual_calibrate(
    config: AppConfig,
    *,
    address: str | None = None,
    preview: bool = False,
    start_delay_ms: float | None = None,
    config_path: Path | None = None,
) -> CalibrateResult:
    app = VisualCalibrator(
        config,
        address=address,
        preview=preview,
        start_delay_ms=start_delay_ms,
        config_path=config_path,
    )
    return app.run()
