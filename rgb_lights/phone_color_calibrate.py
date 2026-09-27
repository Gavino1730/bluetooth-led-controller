from __future__ import annotations

"""Phone color calibrator — match strip colors to your phone screen over Wi‑Fi.

Opens a mobile web UI:
  1. Tap a reference color (fills the phone screen)
  2. Strip jumps to the same color with live sat / R/G/B / brightness
  3. Tweak until the strip matches the phone, then Save → config.toml
"""

import asyncio
import colorsys
import json
import queue
import re
import ssl
import subprocess
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from rgb_lights.ble import BleLightController
from rgb_lights.color_matrix import (
    CAL_PATCHES,
    IDENTITY_9,
    as_matrix9,
    fit_color_matrix,
    is_identity,
    matrix_to_gains,
)
from rgb_lights.config import AppConfig
from rgb_lights.phone_calibrate import lan_ips


DEFAULT_PORT = 8766
CERT_DIR = Path(__file__).resolve().parent.parent / ".certs"

SWATCHES: list[dict[str, Any]] = [
    {"id": pid, "label": pid.capitalize(), "rgb": rgb} for pid, rgb in CAL_PATCHES
] + [
    {"id": "skin", "label": "Warm", "rgb": (210, 150, 110)},
]


@dataclass
class ColorCalState:
    swatch: str = "white"
    saturation: float = 1.35
    brightness: float = 0.85
    r_gain: float = 1.0
    g_gain: float = 1.0
    b_gain: float = 1.0
    color_matrix: list[float] = None  # type: ignore[assignment]
    message: str = "Open this page on your phone"
    saved: bool = False
    connected: bool = False
    preview_rgb: tuple[int, int, int] = (255, 255, 255)
    auto_running: bool = False
    auto_step: str = ""
    fit_error: float = 0.0
    samples: dict[str, list[float]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.color_matrix is None:
            self.color_matrix = list(IDENTITY_9)
        if self.samples is None:
            self.samples = {}


def apply_preview(
    base: tuple[int, int, int],
    *,
    saturation: float,
    brightness: float,
    r_gain: float,
    g_gain: float,
    b_gain: float,
) -> tuple[int, int, int]:
    """Same idea as movie LED match — what the strip will show for this swatch."""
    r = base[0] * max(0.2, r_gain)
    g = base[1] * max(0.2, g_gain)
    b = base[2] * max(0.2, b_gain)
    h, s, v = colorsys.rgb_to_hsv(
        min(1.0, r / 255.0),
        min(1.0, g / 255.0),
        min(1.0, b / 255.0),
    )
    if saturation > 1.0 and 0.04 < s < 0.92:
        s = min(1.0, s + (1.0 - s) * (saturation - 1.0) * 0.55)
    if v > 0.93 and s < 0.10:
        s *= 0.45
    v = max(0.0, min(1.0, v * brightness))
    nr, ng, nb = colorsys.hsv_to_rgb(h, s, v)
    return int(round(nr * 255)), int(round(ng * 255)), int(round(nb * 255))


def write_color_cal_to_config(
    *,
    saturation: float,
    brightness: float,
    r_gain: float,
    g_gain: float,
    b_gain: float,
    color_matrix: list[float] | None = None,
    path: Path | None = None,
) -> Path:
    path = path or Path.cwd() / "config.toml"
    text = path.read_text(encoding="utf-8") if path.exists() else ""

    def set_in_section(src: str, section: str, key: str, value: str) -> str:
        header = f"[{section}]"
        if header not in src:
            src = src.rstrip() + f"\n\n{header}\n"
        chunks = re.split(r"(?m)(?=^\[)", src)
        out: list[str] = []
        for chunk in chunks:
            if chunk.startswith(header):
                pat = rf"(?m)^({re.escape(key)}\s*=\s*).*$"
                if re.search(rf"(?m)^{re.escape(key)}\s*=", chunk):
                    chunk = re.sub(pat, rf"\g<1>{value}", chunk, count=1)
                else:
                    chunk = chunk.rstrip() + f"\n{key} = {value}\n"
            out.append(chunk)
        return "".join(out)

    text = set_in_section(text, "movie", "saturation", f"{saturation:.2f}")
    text = set_in_section(text, "movie", "r_gain", f"{r_gain:.2f}")
    text = set_in_section(text, "movie", "g_gain", f"{g_gain:.2f}")
    text = set_in_section(text, "movie", "b_gain", f"{b_gain:.2f}")
    mat = as_matrix9(color_matrix)
    mat_txt = "[" + ", ".join(f"{v:.4f}" for v in mat) + "]"
    text = set_in_section(text, "movie", "color_matrix", mat_txt)
    text = set_in_section(text, "reactive", "intensity", f"{brightness:.2f}")
    path.write_text(text, encoding="utf-8")
    return path


class ColorCalHub:
    def __init__(self, config: AppConfig, config_path: Path, preview: bool = False) -> None:
        self.config = config
        self.config_path = config_path
        self.preview = preview
        mc = config.movie
        sat = mc.saturation if mc.saturation > 0 else 1.35
        self.state = ColorCalState(
            saturation=float(sat),
            brightness=float(config.reactive.intensity),
            r_gain=float(mc.r_gain),
            g_gain=float(mc.g_gain),
            b_gain=float(mc.b_gain),
            color_matrix=as_matrix9(mc.color_matrix),
        )
        self._lock = threading.Lock()
        self._sse: list[queue.Queue[str]] = []
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, name="color-cal-ble", daemon=True)
        self._controller: BleLightController | None = None
        self._ready = threading.Event()
        self._error: str | None = None

    def start(self) -> None:
        self._thread.start()
        self._ready.wait(timeout=12.0)
        if self._error:
            raise RuntimeError(self._error)

    def stop(self) -> None:
        def _shutdown() -> None:
            async def _close() -> None:
                if self._controller is not None:
                    try:
                        await self._controller.set_rgb(0, 0, 0, force=True)
                        await self._controller.power_off()
                    except Exception:
                        pass
                    await self._controller.disconnect()
                    self._controller = None

            self._loop.create_task(_close())
            self._loop.call_later(0.4, self._loop.stop)

        if self._loop.is_running():
            self._loop.call_soon_threadsafe(_shutdown)
        self._thread.join(timeout=3.0)
        self._broadcast({"type": "bye"})

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.create_task(self._connect())
        self._loop.run_forever()

    async def _connect(self) -> None:
        try:
            if self.preview:
                with self._lock:
                    self.state.connected = True
                    self.state.message = "Preview mode — no BLE"
                self._ready.set()
                self._push_lights()
                return
            ctrl = BleLightController(
                address=self.config.device.address,
                characteristic=self.config.device.characteristic or None,
            )
            await ctrl.connect()
            await ctrl.power_on()
            self._controller = ctrl
            with self._lock:
                self.state.connected = True
                self.state.message = "Connected — pick a color on your phone"
            self._ready.set()
            self._push_lights()
        except Exception as exc:  # noqa: BLE001
            self._error = str(exc)
            with self._lock:
                self.state.message = f"BLE failed: {exc}"
            self._ready.set()

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
                "swatch": s.swatch,
                "saturation": round(s.saturation, 2),
                "brightness": round(s.brightness, 2),
                "r_gain": round(s.r_gain, 2),
                "g_gain": round(s.g_gain, 2),
                "b_gain": round(s.b_gain, 2),
                "color_matrix": [round(x, 4) for x in s.color_matrix],
                "fit_error": round(s.fit_error, 4),
                "message": s.message,
                "saved": s.saved,
                "connected": s.connected,
                "preview_rgb": list(s.preview_rgb),
                "auto_running": s.auto_running,
                "auto_step": s.auto_step,
                "swatches": [
                    {"id": x["id"], "label": x["label"], "rgb": list(x["rgb"])} for x in SWATCHES
                ],
            }

    def update(self, patch: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if "swatch" in patch:
                sid = str(patch["swatch"])
                if any(s["id"] == sid for s in SWATCHES):
                    self.state.swatch = sid
            if "saturation" in patch:
                self.state.saturation = float(np_clip(patch["saturation"], 1.0, 2.2))
            if "brightness" in patch:
                self.state.brightness = float(np_clip(patch["brightness"], 0.25, 1.0))
            if "r_gain" in patch:
                self.state.r_gain = float(np_clip(patch["r_gain"], 0.5, 1.5))
            if "g_gain" in patch:
                self.state.g_gain = float(np_clip(patch["g_gain"], 0.5, 1.5))
            if "b_gain" in patch:
                self.state.b_gain = float(np_clip(patch["b_gain"], 0.5, 1.5))
            self.state.saved = False
            self.state.message = "Live — match strip to phone screen"
        self._push_lights()
        snap = self.snapshot()
        self._broadcast({"type": "state", **snap})
        return snap

    def save(self) -> dict[str, Any]:
        with self._lock:
            s = self.state
            sat, bri = s.saturation, s.brightness
            rg, gg, bg = s.r_gain, s.g_gain, s.b_gain
            mat = list(s.color_matrix)
        path = write_color_cal_to_config(
            saturation=sat,
            brightness=bri,
            r_gain=rg,
            g_gain=gg,
            b_gain=bg,
            color_matrix=mat,
            path=self.config_path,
        )
        with self._lock:
            self.state.saved = True
            self.state.message = f"Saved to {path.name}"
        snap = self.snapshot()
        snap["path"] = str(path)
        self._broadcast({"type": "state", **snap})
        print(f"Saved color cal -> {path}")
        return snap

    def auto_step(self, data: dict[str, Any]) -> dict[str, Any]:
        """Camera-driven calibration step for iPhone full-gamut matrix fit."""
        step = str(data.get("step") or "")
        measured = data.get("measured") or [0, 0, 0]
        ambient = data.get("ambient")
        try:
            mr, mg, mb = (float(measured[0]), float(measured[1]), float(measured[2]))
        except (TypeError, ValueError, IndexError):
            return {"error": "bad measured rgb", **self.snapshot()}

        if ambient is not None:
            try:
                ar, ag, ab = (float(ambient[0]), float(ambient[1]), float(ambient[2]))
                mr = max(0.0, mr - ar * 0.65)
                mg = max(0.0, mg - ag * 0.65)
                mb = max(0.0, mb - ab * 0.65)
            except (TypeError, ValueError, IndexError):
                pass

        patch_ids = {pid for pid, _ in CAL_PATCHES}

        with self._lock:
            self.state.auto_running = True
            self.state.auto_step = step
            self.state.saved = False

            if step == "begin":
                self.state.samples.clear()
                # Reset skewed gains from old calibrator; matrix will do the heavy lifting
                self.state.r_gain = 1.0
                self.state.g_gain = 1.0
                self.state.b_gain = 1.0
                self.state.color_matrix = list(IDENTITY_9)
                self.state.fit_error = 0.0
                self.state.message = "Auto: fill yellow box with the LED strip (iPhone back camera)"
            elif step == "black":
                self.state.message = "Auto: measuring room light..."
            elif step in patch_ids:
                self.state.swatch = step
                self.state.samples[step] = [mr, mg, mb]
                n = len(self.state.samples)
                self.state.message = f"Auto sampled {step} ({n}/{len(CAL_PATCHES)})"
            elif step == "solve":
                targets: list[tuple[float, float, float]] = []
                measures: list[tuple[float, float, float]] = []
                for pid, prgb in CAL_PATCHES:
                    sample = self.state.samples.get(pid)
                    if not sample:
                        continue
                    targets.append((float(prgb[0]), float(prgb[1]), float(prgb[2])))
                    measures.append((float(sample[0]), float(sample[1]), float(sample[2])))
                mat, err = fit_color_matrix(targets, measures)
                self.state.color_matrix = mat
                self.state.fit_error = err
                rg, gg, bg = matrix_to_gains(mat)
                # Keep gains near neutral when matrix is active; slight diag hint for UI
                self.state.r_gain = 1.0
                self.state.g_gain = 1.0
                self.state.b_gain = 1.0
                # Soft sat from fit quality
                if err < 0.08:
                    self.state.saturation = np_clip(self.state.saturation, 1.15, 1.55)
                self.state.message = (
                    f"Matrix fit error={err:.3f} over {len(targets)} patches "
                    f"(diag ~ {rg:.2f}/{gg:.2f}/{bg:.2f})"
                )
            elif step == "done":
                self.state.auto_running = False
                self.state.auto_step = ""
                if is_identity(self.state.color_matrix):
                    self.state.message = "Auto done but matrix weak - retry closer to the strip, then Save"
                else:
                    self.state.message = "Auto tune done - review colors, then Save"
            elif step == "abort":
                self.state.auto_running = False
                self.state.auto_step = ""
                self.state.message = "Auto tune cancelled"
            else:
                self.state.message = f"Unknown auto step: {step}"

        if step == "black":
            self._push_black()
        elif step == "begin" or step in patch_ids:
            if step == "begin":
                with self._lock:
                    self.state.swatch = "white"
            self._push_lights()
        snap = self.snapshot()
        self._broadcast({"type": "state", **snap})
        return snap

    def _push_black(self) -> None:
        with self._lock:
            self.state.preview_rgb = (0, 0, 0)
        if self.preview or self._controller is None:
            return

        async def _write() -> None:
            assert self._controller is not None
            try:
                await self._controller.set_rgb(0, 0, 0, force=True)
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    self.state.message = f"BLE write failed: {exc}"
                self._broadcast({"type": "state", **self.snapshot()})

        self._loop.call_soon_threadsafe(lambda: self._loop.create_task(_write()))

    def _current_base(self) -> tuple[int, int, int]:
        sid = self.state.swatch
        for s in SWATCHES:
            if s["id"] == sid:
                return s["rgb"]  # type: ignore[return-value]
        return (255, 255, 255)

    def _push_lights(self) -> None:
        with self._lock:
            base = self._current_base()
            rgb = apply_preview(
                base,
                saturation=self.state.saturation,
                brightness=self.state.brightness,
                r_gain=self.state.r_gain,
                g_gain=self.state.g_gain,
                b_gain=self.state.b_gain,
            )
            self.state.preview_rgb = rgb

        if self.preview or self._controller is None:
            return

        async def _write() -> None:
            assert self._controller is not None
            try:
                await self._controller.set_rgb(*rgb, force=True)
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    self.state.message = f"BLE write failed: {exc}"
                self._broadcast({"type": "state", **self.snapshot()})

        self._loop.call_soon_threadsafe(lambda: self._loop.create_task(_write()))

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


def np_clip(value: Any, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def ensure_ssl_certs(extra_ips: list[str] | None = None) -> tuple[Path, Path]:
    """Create a local self-signed cert so phones allow the camera (needs HTTPS)."""
    CERT_DIR.mkdir(parents=True, exist_ok=True)
    cert_path = CERT_DIR / "cert.pem"
    key_path = CERT_DIR / "key.pem"
    if cert_path.exists() and key_path.exists():
        return cert_path, key_path

    ips = ["127.0.0.1", *(extra_ips or []), *lan_ips()]
    # de-dupe preserve order
    seen: set[str] = set()
    uniq_ips: list[str] = []
    for ip in ips:
        if ip not in seen:
            seen.add(ip)
            uniq_ips.append(ip)

    try:
        return _write_certs_cryptography(cert_path, key_path, uniq_ips)
    except Exception:
        pass

    # Fallback: openssl CLI
    san = ",".join([f"IP:{ip}" for ip in uniq_ips] + ["DNS:localhost"])
    cmd = [
        "openssl",
        "req",
        "-x509",
        "-newkey",
        "rsa:2048",
        "-keyout",
        str(key_path),
        "-out",
        str(cert_path),
        "-days",
        "825",
        "-nodes",
        "-subj",
        "/CN=RGB Lights Calibrator",
        "-addext",
        f"subjectAltName={san}",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return cert_path, key_path
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Could not create HTTPS certificate for camera access. "
            "Install OpenSSL or: pip install cryptography"
        ) from exc


def _write_certs_cryptography(cert_path: Path, key_path: Path, ips: list[str]) -> tuple[Path, Path]:
    from datetime import datetime, timedelta, timezone

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name(
        [x509.NameAttribute(NameOID.COMMON_NAME, "RGB Lights Calibrator")]
    )
    alt = [x509.DNSName("localhost")]
    for ip in ips:
        try:
            alt.append(x509.IPAddress(__import__("ipaddress").ip_address(ip)))
        except ValueError:
            continue
    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=825))
        .add_extension(x509.SubjectAlternativeName(alt), critical=False)
        .sign(key, hashes.SHA256())
    )
    key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return cert_path, key_path


PAGE_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"/>
<meta name="apple-mobile-web-app-capable" content="yes"/>
<title>RGB color calibrate</title>
<style>
  :root {
    --bg: #0c0f14;
    --panel: #151a22;
    --text: #eef2f8;
    --muted: #8b93a7;
    --accent: #f0c24b;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; min-height: 100dvh; background: var(--bg); color: var(--text);
    font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif;
    display: flex; flex-direction: column;
  }
  #camWrap {
    position: relative; flex: 1; min-height: 42dvh; margin: 12px; border-radius: 18px;
    overflow: hidden; background: #000;
  }
  #cam { width: 100%; height: 100%; object-fit: cover; display: block; min-height: 42dvh; background: #111; }
  #camPlaceholder {
    position: absolute; inset: 0; display: grid; place-items: center; padding: 20px;
    text-align: center; color: #c9d0dc; font-size: .95rem; line-height: 1.45;
    background: #12151c;
  }
  #camPlaceholder.hide { display: none; }
  #reticle {
    position: absolute; left: 50%; top: 50%; width: 34%; height: 22%;
    transform: translate(-50%, -50%); border: 2px solid var(--accent);
    border-radius: 12px; box-shadow: 0 0 0 999px rgba(0,0,0,.28); pointer-events: none;
  }
  #sampleSwatch {
    position: absolute; right: 12px; bottom: 12px; width: 48px; height: 48px;
    border-radius: 10px; border: 2px solid #fff; background: #333;
  }
  #secureBanner {
    display: none; margin: 0 12px 8px; padding: 12px 14px; border-radius: 12px;
    background: #3a2410; color: #ffd9a8; font-size: .9rem; line-height: 1.4;
  }
  #secureBanner.show { display: block; }
  main { padding: 0 14px 22px; }
  h1 { font-size: 1.1rem; margin: 0 0 4px; font-weight: 650; }
  p { margin: 0 0 12px; color: var(--muted); font-size: .9rem; line-height: 1.4; }
  .chips { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-bottom: 12px; }
  .chip {
    appearance: none; border: 2px solid transparent; border-radius: 12px;
    padding: 10px 8px; font-size: .82rem; font-weight: 650; color: #111; min-height: 44px;
  }
  .chip.active { border-color: var(--accent); }
  .chip.dark { color: #f5f5f5; }
  details.manual { margin-top: 8px; color: var(--muted); font-size: .88rem; }
  details.manual summary { cursor: pointer; margin-bottom: 8px; }
  label.row {
    display: grid; grid-template-columns: 92px 1fr 48px; gap: 8px; align-items: center;
    margin: 8px 0; font-size: .85rem; color: var(--muted);
  }
  label.row span.val { font-family: ui-monospace, Consolas, monospace; color: var(--text); text-align: right; }
  input[type=range] { width: 100%; accent-color: var(--accent); }
  .actions { display: flex; gap: 8px; margin-top: 12px; flex-wrap: wrap; }
  button.act {
    appearance: none; border: 0; border-radius: 12px; padding: 14px 16px;
    font-size: 1rem; font-weight: 600; background: var(--panel); color: var(--text); flex: 1; min-width: 120px;
  }
  button.act.primary { background: var(--accent); color: #1a1408; }
  button.act:active { transform: scale(.98); }
  button.act:disabled { opacity: .45; }
  #status { margin-top: 12px; color: var(--muted); font-size: .88rem; min-height: 2.8em; }
  .tag { display: inline-block; padding: 3px 8px; border-radius: 999px; background: #1c2230; font-size: .75rem; color: var(--muted); margin-bottom: 8px; }
  #progress { height: 4px; background: #1c2230; border-radius: 99px; overflow: hidden; margin-top: 10px; display: none; }
  #progress > i { display: block; height: 100%; width: 0%; background: var(--accent); transition: width .25s; }
</style>
</head>
<body>
  <div id="secureBanner"></div>
  <div id="camWrap">
    <video id="cam" playsinline muted autoplay></video>
    <div id="camPlaceholder">Starting camera…</div>
    <div id="reticle"></div>
    <div id="sampleSwatch"></div>
  </div>
  <main>
    <div class="tag" id="connTag">connecting</div>
    <h1>Color calibrate</h1>
    <p><b>iPhone tip:</b> use the back camera, dim the room a bit, and fill the yellow box with only the glowing strip. Auto samples white + RGB + CMY for a full color matrix.</p>
    <div class="actions">
      <button class="act primary" id="auto">Auto tune</button>
      <button class="act" id="enableCam">Enable camera</button>
    </div>
    <div class="actions">
      <button class="act" id="flipCam">Flip camera</button>
      <button class="act" id="save">Save</button>
    </div>
    <div id="progress"><i id="bar"></i></div>
    <div id="status">Connecting…</div>
    <div class="chips" id="chips"></div>
    <details class="manual">
      <summary>Manual sliders</summary>
      <label class="row">Saturation <input id="sat" type="range" min="100" max="220" value="135"/><span class="val" id="satV">1.35</span></label>
      <label class="row">Brightness <input id="bri" type="range" min="25" max="100" value="85"/><span class="val" id="briV">0.85</span></label>
      <label class="row">Red gain <input id="rg" type="range" min="50" max="150" value="100"/><span class="val" id="rgV">1.00</span></label>
      <label class="row">Green gain <input id="gg" type="range" min="50" max="150" value="100"/><span class="val" id="ggV">1.00</span></label>
      <label class="row">Blue gain <input id="bg" type="range" min="50" max="150" value="100"/><span class="val" id="bgV">1.00</span></label>
      <div class="actions">
        <button class="act" id="reset">Reset</button>
        <button class="act" id="abort" disabled>Stop auto</button>
      </div>
    </details>
  </main>
<script>
const $ = (id) => document.getElementById(id);
const statusEl = $("status");
const connTag = $("connTag");
const chipsEl = $("chips");
const video = $("cam");
const sampleSwatch = $("sampleSwatch");
const progress = $("progress");
const bar = $("bar");
const camPlaceholder = $("camPlaceholder");
const secureBanner = $("secureBanner");

let swatches = [];
let current = "white";
let debounce = null;
let autoAbort = false;
let stream = null;
let preferEnv = true;
const canvas = document.createElement("canvas");
const ctx = canvas.getContext("2d", { willReadFrequently: true });

function rgbCss(rgb) { return `rgb(${rgb[0]|0},${rgb[1]|0},${rgb[2]|0})`; }
function isDark(rgb) { return (0.299*rgb[0]+0.587*rgb[1]+0.114*rgb[2]) < 110; }
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function showCamError(msg) {
  camPlaceholder.classList.remove("hide");
  camPlaceholder.textContent = msg;
  statusEl.textContent = msg;
}

function checkSecure() {
  if (window.isSecureContext) return true;
  const httpsUrl = location.href.replace(/^http:/, "https:");
  secureBanner.classList.add("show");
  secureBanner.innerHTML =
    "Camera needs HTTPS on phones. Open <b>" + httpsUrl + "</b> instead " +
    "(accept the certificate warning — Advanced → Proceed).";
  showCamError("Blocked: open the https:// link printed on your PC.");
  return false;
}

function renderChips() {
  chipsEl.innerHTML = "";
  for (const s of swatches) {
    const b = document.createElement("button");
    b.className = "chip" + (isDark(s.rgb) ? " dark" : "") + (s.id === current ? " active" : "");
    b.style.background = rgbCss(s.rgb);
    b.textContent = s.label;
    b.onclick = () => { if (!autoAbort && !$("auto").disabled) pick(s.id); };
    chipsEl.appendChild(b);
  }
}

function applyState(st) {
  if (st.swatches) swatches = st.swatches;
  current = st.swatch || current;
  renderChips();
  $("sat").value = Math.round((st.saturation ?? 1.35) * 100);
  $("bri").value = Math.round((st.brightness ?? 0.85) * 100);
  $("rg").value = Math.round((st.r_gain ?? 1) * 100);
  $("gg").value = Math.round((st.g_gain ?? 1) * 100);
  $("bg").value = Math.round((st.b_gain ?? 1) * 100);
  $("satV").textContent = (st.saturation ?? 1.35).toFixed(2);
  $("briV").textContent = (st.brightness ?? 0.85).toFixed(2);
  $("rgV").textContent = (st.r_gain ?? 1).toFixed(2);
  $("ggV").textContent = (st.g_gain ?? 1).toFixed(2);
  $("bgV").textContent = (st.b_gain ?? 1).toFixed(2);
  connTag.textContent = st.connected ? "strip connected" : "preview / offline";
  if (!autoRunningUi) statusEl.textContent = st.message || "";
}

let autoRunningUi = false;

async function post(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body || {}),
  });
  return res.json();
}

function pick(id) {
  current = id;
  push({swatch: id});
}

function push(extra) {
  const body = Object.assign({
    swatch: current,
    saturation: Number($("sat").value) / 100,
    brightness: Number($("bri").value) / 100,
    r_gain: Number($("rg").value) / 100,
    g_gain: Number($("gg").value) / 100,
    b_gain: Number($("bg").value) / 100,
  }, extra || {});
  $("satV").textContent = body.saturation.toFixed(2);
  $("briV").textContent = body.brightness.toFixed(2);
  $("rgV").textContent = body.r_gain.toFixed(2);
  $("ggV").textContent = body.g_gain.toFixed(2);
  $("bgV").textContent = body.b_gain.toFixed(2);
  clearTimeout(debounce);
  debounce = setTimeout(async () => {
    try { applyState(await post("/api/update", body)); }
    catch (e) { statusEl.textContent = "Update failed — check Wi‑Fi"; }
  }, 40);
}

async function ensureCamera() {
  if (!checkSecure()) return false;
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showCamError("This browser blocks camera APIs. Try Chrome or Safari.");
    return false;
  }
  if (stream && video.videoWidth) {
    camPlaceholder.classList.add("hide");
    return true;
  }
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  camPlaceholder.classList.remove("hide");
  camPlaceholder.textContent = "Requesting camera…";
  const attempts = [
    { audio: false, video: { facingMode: { ideal: preferEnv ? "environment" : "user" }, width: { ideal: 1280 }, height: { ideal: 720 } } },
    { audio: false, video: { facingMode: preferEnv ? "environment" : "user" } },
    { audio: false, video: true },
  ];
  let lastErr = null;
  for (const constraints of attempts) {
    try {
      stream = await navigator.mediaDevices.getUserMedia(constraints);
      video.setAttribute("playsinline", "true");
      video.muted = true;
      video.srcObject = stream;
      await video.play();
      camPlaceholder.classList.add("hide");
      statusEl.textContent = "Camera on — fill the yellow box with the strip";
      return true;
    } catch (e) {
      lastErr = e;
    }
  }
  const name = lastErr && (lastErr.name || "");
  if (name === "NotAllowedError" || name === "PermissionDeniedError") {
    showCamError("Camera permission denied. Allow camera for this site in browser settings, then tap Enable camera.");
  } else if (name === "NotFoundError" || name === "DevicesNotFoundError") {
    showCamError("No camera found on this phone.");
  } else if (name === "NotReadableError" || name === "TrackStartError") {
    showCamError("Camera busy (another app). Close Instagram/Snapchat/etc, then tap Enable camera.");
  } else {
    showCamError("Camera failed: " + (lastErr && lastErr.message ? lastErr.message : String(lastErr)));
  }
  return false;
}

function sampleCenter() {
  if (!video.videoWidth) return [0,0,0];
  const w = video.videoWidth, h = video.videoHeight;
  canvas.width = w; canvas.height = h;
  ctx.drawImage(video, 0, 0, w, h);
  const rw = Math.floor(w * 0.34), rh = Math.floor(h * 0.22);
  const x = Math.floor((w - rw) / 2), y = Math.floor((h - rh) / 2);
  const data = ctx.getImageData(x, y, rw, rh).data;
  // Prefer brightest pixels in the box (LED strip) - ignores darker wall/phone bezel
  const pix = [];
  for (let i = 0; i < data.length; i += 16) {
    const r = data[i], g = data[i+1], b = data[i+2];
    const yv = 0.299*r + 0.587*g + 0.114*b;
    pix.push([yv, r, g, b]);
  }
  if (!pix.length) return [0,0,0];
  pix.sort((a,b) => b[0]-a[0]);
  const keep = Math.max(8, Math.floor(pix.length * 0.35));
  let r=0,g=0,b=0;
  for (let i = 0; i < keep; i++) { r += pix[i][1]; g += pix[i][2]; b += pix[i][3]; }
  const out = [r/keep, g/keep, b/keep];
  sampleSwatch.style.background = rgbCss(out);
  return out;
}

async function sampleStable(frames = 12) {
  // Median-ish average; drop darkest/brightest frame for iPhone AE flicker
  const samples = [];
  for (let i = 0; i < frames; i++) {
    await sleep(70);
    samples.push(sampleCenter());
  }
  samples.sort((a,b) => (a[0]+a[1]+a[2]) - (b[0]+b[1]+b[2]));
  const use = samples.slice(2, Math.max(3, samples.length - 2));
  let r=0,g=0,b=0;
  for (const s of use) { r += s[0]; g += s[1]; b += s[2]; }
  return [r/use.length, g/use.length, b/use.length];
}

function setProgress(pct) {
  progress.style.display = "block";
  bar.style.width = Math.max(0, Math.min(100, pct)) + "%";
}

async function runAuto() {
  if (!(await ensureCamera())) return;
  autoAbort = false;
  autoRunningUi = true;
  $("auto").disabled = true;
  $("abort").disabled = false;
  setProgress(0);
  statusEl.textContent = "Auto: hold steady - iPhone needs a moment per color…";

  const patchOrder = ["white","red","green","blue","cyan","magenta","yellow","orange"];

  try {
    await post("/api/auto-step", { step: "begin" });
    await sleep(500);

    statusEl.textContent = "Auto: measuring room light…";
    await post("/api/auto-step", { step: "black" });
    await sleep(700);
    const ambient = await sampleStable(14);
    setProgress(6);
    if (autoAbort) throw new Error("abort");

    // Two passes: iPhone AE settles better on second look
    const passes = 2;
    let stepN = 0;
    const total = passes * patchOrder.length;

    for (let p = 0; p < passes; p++) {
      for (const color of patchOrder) {
        if (autoAbort) throw new Error("abort");
        statusEl.textContent = `Pass ${p+1}/${passes}: ${color} — keep strip in the box`;
        await post("/api/update", {
          swatch: color,
          saturation: Number($("sat").value)/100,
          brightness: Number($("bri").value)/100,
          r_gain: 1,
          g_gain: 1,
          b_gain: 1,
        });
        // iPhone auto-exposure settle time
        await sleep(900);
        const measured = await sampleStable(14);
        const st = await post("/api/auto-step", { step: color, measured, ambient });
        applyState(st);
        stepN++;
        setProgress(6 + (stepN / total) * 84);
      }
    }

    statusEl.textContent = "Solving full color matrix…";
    const solved = await post("/api/auto-step", { step: "solve" });
    applyState(solved);
    setProgress(96);

    const done = await post("/api/auto-step", { step: "done" });
    applyState(done);
    setProgress(100);
    statusEl.textContent = "Auto tune finished — check a few colors, then Save";
  } catch (e) {
    if (String(e.message) === "abort") {
      await post("/api/auto-step", { step: "abort" });
      statusEl.textContent = "Auto tune stopped";
    } else {
      statusEl.textContent = "Auto failed: " + (e.message || e);
    }
  } finally {
    autoRunningUi = false;
    $("auto").disabled = false;
    $("abort").disabled = true;
    setTimeout(() => { progress.style.display = "none"; }, 900);
  }
}

["sat","bri","rg","gg","bg"].forEach(id => {
  $(id).addEventListener("input", () => push());
});

$("save").onclick = async () => {
  try { applyState(await post("/api/save", {})); }
  catch (e) { statusEl.textContent = "Save failed"; }
};

$("reset").onclick = () => {
  $("sat").value = 135; $("bri").value = 85;
  $("rg").value = 100; $("gg").value = 100; $("bg").value = 100;
  push({saturation:1.35, brightness:0.85, r_gain:1, g_gain:1, b_gain:1});
};

$("auto").onclick = () => runAuto();
$("abort").onclick = () => { autoAbort = true; };
$("enableCam").onclick = () => ensureCamera();
$("flipCam").onclick = async () => {
  preferEnv = !preferEnv;
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  await ensureCamera();
};

// Live sample preview
setInterval(() => { if (stream && video.videoWidth) sampleCenter(); }, 200);

const es = new EventSource("/api/events");
es.onmessage = (ev) => {
  try {
    if (!autoRunningUi) applyState(JSON.parse(ev.data));
    else {
      const st = JSON.parse(ev.data);
      // still sync slider numbers during auto
      $("sat").value = Math.round((st.saturation ?? 1.35) * 100);
      $("bri").value = Math.round((st.brightness ?? 0.85) * 100);
      $("rg").value = Math.round((st.r_gain ?? 1) * 100);
      $("gg").value = Math.round((st.g_gain ?? 1) * 100);
      $("bg").value = Math.round((st.b_gain ?? 1) * 100);
      $("satV").textContent = (st.saturation ?? 1.35).toFixed(2);
      $("briV").textContent = (st.brightness ?? 0.85).toFixed(2);
      $("rgV").textContent = (st.r_gain ?? 1).toFixed(2);
      $("ggV").textContent = (st.g_gain ?? 1).toFixed(2);
      $("bgV").textContent = (st.b_gain ?? 1).toFixed(2);
      if (st.message) statusEl.textContent = st.message;
    }
  } catch (_) {}
};
es.onerror = () => { statusEl.textContent = "Reconnecting…"; };

fetch("/api/state").then(r => r.json()).then(applyState).catch(() => {
  statusEl.textContent = "Could not load state";
});
ensureCamera();
</script>
</body>
</html>
"""


def _make_handler(hub: ColorCalHub):  # noqa: ANN202
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
            return

        def _json(self, code: int, payload: dict[str, Any]) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                raw = PAGE_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return
            if path == "/api/state":
                self._json(200, hub.snapshot())
                return
            if path == "/api/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                q = hub.subscribe()
                try:
                    while True:
                        try:
                            line = q.get(timeout=15.0)
                            self.wfile.write(line.encode("utf-8"))
                            self.wfile.flush()
                        except queue.Empty:
                            self.wfile.write(b": ping\n\n")
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
                if path == "/api/update":
                    self._json(200, hub.update(data))
                    return
                if path == "/api/auto-step":
                    self._json(200, hub.auto_step(data))
                    return
                if path == "/api/save":
                    self._json(200, hub.save())
                    return
            except Exception as exc:  # noqa: BLE001
                self._json(500, {"error": str(exc)})
                return
            self._json(404, {"error": "not found"})

    return Handler


def run_phone_color_calibrate(
    config: AppConfig,
    *,
    host: str = "0.0.0.0",
    port: int = DEFAULT_PORT,
    preview: bool = False,
    https: bool = True,
    config_path: Path | None = None,
) -> int:
    path = config_path or Path.cwd() / "config.toml"
    hub = ColorCalHub(config, path, preview=preview)
    try:
        hub.start()
    except Exception as exc:  # noqa: BLE001
        print(f"Color calibrator failed to start: {exc}")
        return 1

    handler = _make_handler(hub)
    server = ThreadingHTTPServer((host, port), handler)

    scheme = "http"
    if https:
        try:
            cert_path, key_path = ensure_ssl_certs()
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(certfile=str(cert_path), keyfile=str(key_path))
            server.socket = ctx.wrap_socket(server.socket, server_side=True)
            scheme = "https"
        except Exception as exc:  # noqa: BLE001
            print(f"HTTPS setup failed ({exc}) - falling back to HTTP.")
            print("Phone cameras usually NEED https:// - install: pip install cryptography\n")

    urls = [f"{scheme}://127.0.0.1:{port}"]
    for ip in lan_ips():
        urls.append(f"{scheme}://{ip}:{port}")

    print("Phone color calibrator", flush=True)
    print("======================", flush=True)
    print("1. PC + phone on the same Wi-Fi", flush=True)
    print("2. On your phone, open (use HTTPS for camera):", flush=True)
    for u in urls:
        print(f"   {u}", flush=True)
    if scheme == "https":
        print("3. Accept the certificate warning (Advanced -> Proceed / Trust).", flush=True)
        print("4. Allow camera. Fill the yellow box with the LED strip, tap Auto tune, then Save.", flush=True)
    else:
        print("3. WARNING: without HTTPS, phone browsers block the camera.", flush=True)
    if preview:
        print("Preview mode - lights not written.", flush=True)
    print("Ctrl+C to stop.\n", flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        server.shutdown()
        hub.stop()
    return 0
