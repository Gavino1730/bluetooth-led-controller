from __future__ import annotations

"""Calibrate Bluetooth speaker delay using your phone on the same Wi‑Fi.

Opens a local web UI on your phone:
  - Manual: PC beeps, phone screen flashes after delay_ms — tune until they match
  - Auto: phone mic hears the beeps and measures lag (needs mic permission;
    some phones only allow mic on HTTPS — manual mode always works)
"""

import json
import queue
import socket
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

from rgb_lights.calibrate import write_delay_to_config
from rgb_lights.config import AppConfig


SAMPLE_RATE = 44100
BEEP_GAP_S = 1.1
DEFAULT_PORT = 8765


def lan_ips() -> list[str]:
    ips: list[str] = []
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127.") and ip not in ips:
                ips.append(ip)
    except OSError:
        pass
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        sock.close()
        if ip not in ips and not ip.startswith("127."):
            ips.insert(0, ip)
    except OSError:
        pass
    return ips


def _beep_pcm() -> bytes:
    duration = 0.055
    n = int(SAMPLE_RATE * duration)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    tone = 0.9 * np.sin(2 * np.pi * 880.0 * t) * np.exp(-t * 28.0)
    pad = np.zeros(int(0.015 * SAMPLE_RATE), dtype=np.float32)
    audio = np.concatenate([pad, tone])
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


class TonePlayer:
    def __init__(self) -> None:
        try:
            import pyaudiowpatch as pyaudio
        except ImportError as exc:
            raise RuntimeError("PyAudioWPatch is required on Windows") from exc
        self._pyaudio = pyaudio
        self._pa = pyaudio.PyAudio()
        self._pcm = _beep_pcm()
        self._lock = threading.Lock()

    def play(self) -> None:
        def _worker() -> None:
            with self._lock:
                stream = self._pa.open(
                    format=self._pyaudio.paInt16,
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

        threading.Thread(target=_worker, name="phone-tone", daemon=True).start()

    def close(self) -> None:
        with self._lock:
            self._pa.terminate()


@dataclass
class SessionState:
    delay_ms: float
    running: bool = False
    mode: str = "manual"  # manual | auto
    message: str = "Open this page on your phone"
    clicks_done: int = 0
    clicks_total: int = 6
    samples_ms: list[float] = field(default_factory=list)
    measured_ms: float | None = None
    saved: bool = False
    pending_clicks: dict[int, float] = field(default_factory=dict)
    clock_offset_ms: float = 0.0  # server = client + offset
    last_rtt_ms: float = 0.0


class CalibrateHub:
    def __init__(self, delay_ms: float, config_path: Path) -> None:
        self.config_path = config_path
        self.state = SessionState(delay_ms=float(np.clip(delay_ms, 0.0, 600.0)))
        self._player: TonePlayer | None = None
        self._lock = threading.Lock()
        self._sse: list[queue.Queue[str]] = []
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._loop, name="phone-cal-loop", daemon=True)
        self._click_id = 0

    def _get_player(self) -> TonePlayer:
        if self._player is None:
            self._player = TonePlayer()
        return self._player

    def start(self) -> None:
        self._worker.start()

    def stop(self) -> None:
        self._stop.set()
        self._broadcast({"type": "bye"})
        self._worker.join(timeout=2.0)
        if self._player is not None:
            self._player.close()
            self._player = None

    def subscribe(self) -> queue.Queue[str]:
        q: queue.Queue[str] = queue.Queue(maxsize=64)
        with self._lock:
            self._sse.append(q)
        q.put(self._sse_pack({"type": "hello", **self.snapshot()}))
        return q

    def unsubscribe(self, q: queue.Queue[str]) -> None:
        with self._lock:
            if q in self._sse:
                self._sse.remove(q)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            s = self.state
            return {
                "delay_ms": int(round(s.delay_ms)),
                "running": s.running,
                "mode": s.mode,
                "message": s.message,
                "clicks_done": s.clicks_done,
                "clicks_total": s.clicks_total,
                "samples_ms": [round(x, 1) for x in s.samples_ms],
                "measured_ms": None if s.measured_ms is None else int(round(s.measured_ms)),
                "saved": s.saved,
                "rtt_ms": round(s.last_rtt_ms, 1),
            }

    def set_delay(self, delay_ms: float) -> dict[str, Any]:
        with self._lock:
            self.state.delay_ms = float(np.clip(delay_ms, 0.0, 600.0))
            self.state.saved = False
            self.state.message = f"delay_ms = {int(round(self.state.delay_ms))}"
        snap = self.snapshot()
        self._broadcast({"type": "state", **snap})
        return snap

    def save(self) -> dict[str, Any]:
        with self._lock:
            delay = self.state.delay_ms
        path = write_delay_to_config(delay, self.config_path)
        with self._lock:
            self.state.saved = True
            self.state.message = f"Saved delay_ms = {int(round(delay))}"
        snap = self.snapshot()
        snap["path"] = str(path)
        self._broadcast({"type": "state", **snap})
        print(f"Saved delay_ms = {int(round(delay))} to {path}")
        return snap

    def start_manual(self) -> dict[str, Any]:
        with self._lock:
            self.state.mode = "manual"
            self.state.running = True
            self.state.message = "Match phone flash to the beep"
            self.state.samples_ms.clear()
            self.state.measured_ms = None
            self.state.saved = False
        snap = self.snapshot()
        self._broadcast({"type": "state", **snap})
        return snap

    def start_auto(self, clicks: int = 6) -> dict[str, Any]:
        with self._lock:
            self.state.mode = "auto"
            self.state.running = True
            self.state.clicks_total = max(3, min(12, int(clicks)))
            self.state.clicks_done = 0
            self.state.samples_ms.clear()
            self.state.measured_ms = None
            self.state.pending_clicks.clear()
            self.state.saved = False
            self.state.message = "Listening via phone mic..."
        snap = self.snapshot()
        self._broadcast({"type": "state", **snap})
        return snap

    def stop_run(self) -> dict[str, Any]:
        with self._lock:
            self.state.running = False
            self.state.message = "Stopped"
        snap = self.snapshot()
        self._broadcast({"type": "state", **snap})
        return snap

    def sync_clock(self, client_send_ms: float) -> dict[str, Any]:
        server_ms = time.time() * 1000.0
        return {"server_ms": server_ms, "echo_client_ms": client_send_ms}

    def apply_sync(self, client_send_ms: float, client_recv_ms: float, server_ms: float) -> dict[str, Any]:
        # NTP-style: offset so server_time ≈ client_time + offset
        offset = server_ms - (client_send_ms + client_recv_ms) / 2.0
        rtt = max(0.0, client_recv_ms - client_send_ms)
        with self._lock:
            self.state.clock_offset_ms = offset
            self.state.last_rtt_ms = rtt
        return {"offset_ms": offset, "rtt_ms": rtt}

    def report_detect(self, click_id: int, client_peak_ms: float) -> dict[str, Any]:
        with self._lock:
            play_ms = self.state.pending_clicks.pop(click_id, None)
            offset = self.state.clock_offset_ms
            if play_ms is None:
                return {"ok": False, "error": "unknown click id"}
            peak_server = client_peak_ms + offset
            lag = peak_server - play_ms
            if 40.0 <= lag <= 700.0:
                self.state.samples_ms.append(lag)
                self.state.clicks_done = len(self.state.samples_ms)
                self.state.message = f"Click {self.state.clicks_done}/{self.state.clicks_total}: {lag:.0f} ms"
                if len(self.state.samples_ms) >= self.state.clicks_total:
                    self.state.measured_ms = float(np.median(self.state.samples_ms))
                    self.state.delay_ms = float(np.clip(self.state.measured_ms, 0.0, 600.0))
                    self.state.running = False
                    self.state.message = (
                        f"Measured {int(round(self.state.measured_ms))} ms - tap Save"
                    )
            else:
                self.state.message = f"Ignored outlier ({lag:.0f} ms)"
        snap = self.snapshot()
        self._broadcast({"type": "state", **snap})
        return {"ok": True, **snap}

    def _loop(self) -> None:
        next_beep = 0.0
        while not self._stop.is_set():
            now = time.perf_counter()
            with self._lock:
                running = self.state.running
                mode = self.state.mode
                delay_s = self.state.delay_ms / 1000.0
                auto_done = (
                    mode == "auto"
                    and len(self.state.samples_ms) >= self.state.clicks_total
                )
            if running and not auto_done and now >= next_beep:
                self._emit_beep(delay_s, mode)
                next_beep = time.perf_counter() + BEEP_GAP_S
            time.sleep(0.01)

    def _emit_beep(self, delay_s: float, mode: str) -> None:
        self._click_id += 1
        click_id = self._click_id
        server_ms = time.time() * 1000.0
        try:
            player = self._get_player()
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self.state.running = False
                self.state.message = f"Audio failed: {exc}"
            self._broadcast({"type": "state", **self.snapshot()})
            print(f"Audio failed: {exc}", flush=True)
            return

        if mode == "auto":
            with self._lock:
                self.state.pending_clicks[click_id] = server_ms
            # Arm phone first so the mic is watching before the speaker plays.
            self._broadcast(
                {
                    "type": "click",
                    "id": click_id,
                    "server_ms": server_ms,
                }
            )
            time.sleep(0.08)
            # Play time is what we compare against (after arming).
            play_ms = time.time() * 1000.0
            with self._lock:
                self.state.pending_clicks[click_id] = play_ms
            player.play()
            return

        # Manual: beep now, flash phone after delay_ms
        player.play()
        self._broadcast({"type": "beep", "id": click_id})
        threading.Timer(
            delay_s,
            lambda: self._broadcast({"type": "flash", "id": click_id}),
        ).start()

    def _broadcast(self, payload: dict[str, Any]) -> None:
        line = self._sse_pack(payload)
        with self._lock:
            targets = list(self._sse)
        dead: list[queue.Queue[str]] = []
        for q in targets:
            try:
                q.put_nowait(line)
            except queue.Full:
                dead.append(q)
        for q in dead:
            self.unsubscribe(q)

    @staticmethod
    def _sse_pack(payload: dict[str, Any]) -> str:
        return f"data: {json.dumps(payload)}\n\n"


PAGE_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"/>
<meta name="apple-mobile-web-app-capable" content="yes"/>
<title>RGB delay calibrate</title>
<style>
  :root {
    --bg: #0c0f14;
    --panel: #151a22;
    --text: #eef2f8;
    --muted: #8b93a7;
    --accent: #f0c24b;
    --flash: #f5f0de;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; min-height: 100dvh; background: var(--bg); color: var(--text);
    font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif;
    display: flex; flex-direction: column;
  }
  #flash {
    flex: 1; min-height: 42dvh; margin: 14px; border-radius: 18px;
    background: #12151c; display: grid; place-items: center;
    transition: background .04s linear;
  }
  #flash.on { background: var(--flash); color: #1a1408; }
  #flash span { font-size: clamp(2rem, 8vw, 3.4rem); font-weight: 700; letter-spacing: .04em; color: #3a4254; }
  #flash.on span { color: #1a1408; }
  main { padding: 0 16px 24px; }
  h1 { font-size: 1.15rem; margin: 0 0 6px; font-weight: 650; }
  p { margin: 0 0 14px; color: var(--muted); font-size: .95rem; line-height: 1.4; }
  .delay {
    font-family: ui-monospace, Consolas, monospace; font-size: 1.75rem; margin-bottom: 8px;
  }
  input[type=range] { width: 100%; accent-color: var(--accent); }
  .row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 14px; }
  button {
    appearance: none; border: 0; border-radius: 12px; padding: 14px 16px;
    font-size: 1rem; font-weight: 600; background: var(--panel); color: var(--text);
    flex: 1; min-width: 120px;
  }
  button.primary { background: var(--accent); color: #1a1408; }
  button:active { transform: scale(.98); }
  #status { margin-top: 14px; color: var(--muted); font-size: .9rem; min-height: 2.6em; }
  .tag { display: inline-block; padding: 3px 8px; border-radius: 999px; background: #1c2230; font-size: .75rem; color: var(--muted); margin-bottom: 10px; }
</style>
</head>
<body>
  <div id="flash"><span>FLASH</span></div>
  <main>
    <div class="tag" id="modeTag">manual</div>
    <h1>Speaker delay</h1>
    <p>PC plays a beep on your Bluetooth speaker. This screen flashes after the delay. Match them, then Save.</p>
    <div class="delay" id="delayLabel">delay_ms = 200</div>
    <input id="slider" type="range" min="0" max="500" value="200"/>
    <div class="row">
      <button id="minus">−10</button>
      <button id="plus">+10</button>
    </div>
    <div class="row">
      <button class="primary" id="manual">Start manual</button>
      <button id="auto">Auto (mic)</button>
    </div>
    <div class="row">
      <button id="save">Save</button>
      <button id="stop">Stop</button>
    </div>
    <div id="status">Connecting…</div>
  </main>
<script>
const $ = (id) => document.getElementById(id);
const flashEl = $("flash");
const statusEl = $("status");
const slider = $("slider");
const delayLabel = $("delayLabel");
const modeTag = $("modeTag");

let delayMs = 200;
let audioCtx = null;
let processor = null;
let mediaStream = null;
let armed = null; // {id, tMsg}
let listening = false;

function setStatus(t) { statusEl.textContent = t; }
function setDelayUI(ms) {
  delayMs = Math.round(ms);
  slider.value = String(delayMs);
  delayLabel.textContent = `delay_ms = ${delayMs}`;
}
function applyState(s) {
  if (s.delay_ms != null) setDelayUI(s.delay_ms);
  if (s.mode) modeTag.textContent = s.mode;
  if (s.message) setStatus(s.message);
  if (s.measured_ms != null) setDelayUI(s.measured_ms);
}

async function post(path, body={}) {
  const r = await fetch(path, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body),
  });
  const data = await r.json();
  if (!r.ok) throw new Error(data.error || r.statusText);
  return data;
}

function flash() {
  flashEl.classList.add("on");
  flashEl.querySelector("span").textContent = "NOW";
  setTimeout(() => {
    flashEl.classList.remove("on");
    flashEl.querySelector("span").textContent = "FLASH";
  }, 90);
}

function connectSSE() {
  const es = new EventSource("/events");
  es.onmessage = (ev) => {
    let msg;
    try { msg = JSON.parse(ev.data); } catch { return; }
    if (msg.type === "hello" || msg.type === "state") applyState(msg);
    if (msg.type === "flash") flash();
    if (msg.type === "beep") { /* audio is on PC */ }
    if (msg.type === "click") onClickEvent(msg);
  };
  es.onerror = () => setStatus("Reconnecting… keep this page open");
}

async function syncClocks() {
  let last = null;
  for (let i = 0; i < 8; i++) {
    const send = Date.now();
    const res = await post("/api/sync", { client_send_ms: send });
    const recv = Date.now();
    last = await post("/api/sync_apply", {
      client_send_ms: send,
      client_recv_ms: recv,
      server_ms: res.server_ms,
    });
    await new Promise(r => setTimeout(r, 40));
  }
  setStatus(`Clock sync ok · RTT ~${Math.round(last.rtt_ms)} ms`);
  return last;
}

async function ensureMic() {
  if (listening) return true;
  if (!window.isSecureContext && location.hostname !== "localhost" && location.hostname !== "127.0.0.1") {
    setStatus("Phone mic needs HTTPS on many phones. Use Start manual instead (works on HTTP).");
    return false;
  }
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      video: false,
    });
  } catch (e) {
    setStatus("Mic permission denied — use Start manual.");
    return false;
  }
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const source = audioCtx.createMediaStreamSource(mediaStream);
  processor = audioCtx.createScriptProcessor(1024, 1, 1);
  let cooldownUntil = 0;
  processor.onaudioprocess = (e) => {
    if (!armed) return;
    const now = Date.now();
    if (now < cooldownUntil) return;
    const input = e.inputBuffer.getChannelData(0);
    let peak = 0;
    for (let i = 0; i < input.length; i++) {
      const v = Math.abs(input[i]);
      if (v > peak) peak = v;
    }
    if (peak >= 0.12) {
      const id = armed.id;
      armed = null;
      cooldownUntil = now + 280;
      post("/api/detect", { id, client_peak_ms: Date.now() }).catch(() => {});
    }
  };
  source.connect(processor);
  processor.connect(audioCtx.destination);
  listening = true;
  return true;
}

function onClickEvent(msg) {
  armed = { id: msg.id, tMsg: Date.now() };
  // Safety: clear if no peak
  setTimeout(() => {
    if (armed && armed.id === msg.id) armed = null;
  }, 900);
}

slider.addEventListener("input", () => {
  setDelayUI(Number(slider.value));
});
slider.addEventListener("change", async () => {
  applyState(await post("/api/delay", { delay_ms: delayMs }));
});
$("minus").onclick = async () => applyState(await post("/api/delay", { delay_ms: delayMs - 10 }));
$("plus").onclick = async () => applyState(await post("/api/delay", { delay_ms: delayMs + 10 }));
$("manual").onclick = async () => applyState(await post("/api/start_manual"));
$("stop").onclick = async () => applyState(await post("/api/stop"));
$("save").onclick = async () => {
  const s = await post("/api/save");
  applyState(s);
  setStatus(s.message || `Saved ${s.delay_ms} ms`);
};
$("auto").onclick = async () => {
  const ok = await ensureMic();
  if (!ok) return;
  await syncClocks();
  applyState(await post("/api/start_auto", { clicks: 6 }));
};

connectSSE();
setStatus("Connected — Start manual, or Auto (mic) if allowed");
</script>
</body>
</html>
"""


def _make_handler(hub: CalibrateHub):  # noqa: ANN001
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: object) -> None:  # noqa: A003
            return

        def _cors(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def _json(self, code: int, payload: dict[str, Any]) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self._cors()
            self.end_headers()
            self.wfile.write(raw)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0") or 0)
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            if not raw:
                return {}
            return json.loads(raw.decode("utf-8"))

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(204)
            self._cors()
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                body = PAGE_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/api/state":
                self._json(200, hub.snapshot())
                return
            if path == "/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self._cors()
                self.end_headers()
                q = hub.subscribe()
                try:
                    while True:
                        try:
                            line = q.get(timeout=15.0)
                        except queue.Empty:
                            self.wfile.write(b": ping\n\n")
                            self.wfile.flush()
                            continue
                        self.wfile.write(line.encode("utf-8"))
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                finally:
                    hub.unsubscribe(q)
                return
            self._json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            try:
                data = self._read_json()
            except json.JSONDecodeError:
                self._json(400, {"error": "invalid json"})
                return
            try:
                if path == "/api/delay":
                    self._json(200, hub.set_delay(float(data.get("delay_ms", hub.state.delay_ms))))
                    return
                if path == "/api/save":
                    self._json(200, hub.save())
                    return
                if path == "/api/start_manual":
                    self._json(200, hub.start_manual())
                    return
                if path == "/api/start_auto":
                    self._json(200, hub.start_auto(int(data.get("clicks", 6))))
                    return
                if path == "/api/stop":
                    self._json(200, hub.stop_run())
                    return
                if path == "/api/sync":
                    self._json(200, hub.sync_clock(float(data.get("client_send_ms", 0.0))))
                    return
                if path == "/api/sync_apply":
                    self._json(
                        200,
                        hub.apply_sync(
                            float(data["client_send_ms"]),
                            float(data["client_recv_ms"]),
                            float(data["server_ms"]),
                        ),
                    )
                    return
                if path == "/api/detect":
                    self._json(
                        200,
                        hub.report_detect(int(data["id"]), float(data["client_peak_ms"])),
                    )
                    return
            except Exception as exc:  # noqa: BLE001
                self._json(500, {"error": str(exc)})
                return
            self._json(404, {"error": "not found"})

    return Handler


def run_phone_calibrate(
    config: AppConfig,
    *,
    host: str = "0.0.0.0",
    port: int = DEFAULT_PORT,
    start_delay_ms: float | None = None,
    config_path: Path | None = None,
) -> int:
    delay = float(
        start_delay_ms if start_delay_ms is not None else config.reactive.delay_ms
    )
    path = config_path or Path.cwd() / "config.toml"
    hub = CalibrateHub(delay_ms=delay, config_path=path)
    handler = _make_handler(hub)
    server = ThreadingHTTPServer((host, port), handler)
    hub.start()

    urls = [f"http://127.0.0.1:{port}"]
    for ip in lan_ips():
        urls.append(f"http://{ip}:{port}")

    print("Phone delay calibrator", flush=True)
    print("======================", flush=True)
    print("1. PC + phone on the same Wi-Fi", flush=True)
    print("2. On your phone, open:", flush=True)
    for u in urls:
        print(f"   {u}", flush=True)
    print("3. Tap Start manual (best). Match flash to beep, then Save.", flush=True)
    print("   Auto (mic) works on some phones; otherwise use manual.", flush=True)
    print("Ctrl+C to stop.\n", flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        server.shutdown()
        hub.stop()
    return 0
