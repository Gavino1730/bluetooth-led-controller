from __future__ import annotations

import argparse
import asyncio
import logging
import time
from pathlib import Path

import numpy as np

from rgb_lights.audio import AudioAnalyzer, AudioCapture, list_audio_devices
from rgb_lights.audio.analyzer import SpectrumBands
from rgb_lights.ble import BleLightController, explore_services, print_scan
from rgb_lights.config import AppConfig, load_config
from rgb_lights.lighting import LightingPipeline
from rgb_lights.reactive import ColorMapper, RateLimiter
from rgb_lights.sync import OutputDelay


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rgb-lights",
        description="Daybetter BLE RGB light controller with music-reactive modes.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.toml (default: ./config.toml)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")

    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="Discover nearby Daybetter BLE devices")
    scan.add_argument("--timeout", type=float, default=5.0)
    scan.add_argument(
        "--all",
        action="store_true",
        help="Show all BLE devices (default is Daybetter service only)",
    )

    explore = sub.add_parser("explore", help="List GATT services/characteristics for a device")
    explore.add_argument("address", nargs="?", default=None, help="BLE address (or use config)")

    on_p = sub.add_parser("on", help="Turn lights on")
    off_p = sub.add_parser("off", help="Turn lights off")
    for p in (on_p, off_p):
        p.add_argument("--address", default=None)

    color = sub.add_parser("color", help="Set a solid RGB color")
    color.add_argument("r", type=int)
    color.add_argument("g", type=int)
    color.add_argument("b", type=int)
    color.add_argument("--address", default=None)

    bright = sub.add_parser("brightness", help="Set brightness 0-100")
    bright.add_argument("level", type=int)
    bright.add_argument("--address", default=None)

    music = sub.add_parser("music", help="React to PC audio / microphone in real time")
    music.add_argument("--address", default=None)
    music.add_argument(
        "--mode",
        choices=[
            "spectrum",
            "bass_pulse",
            "energy",
            "party",
            "calm",
            "darkside",
            "festival",
            "party2",
            "festival2",
        ],
        default=None,
    )
    music.add_argument("--source", choices=["system", "mic"], default=None)
    music.add_argument("--gain", type=float, default=None)
    music.add_argument(
        "--delay-ms",
        type=float,
        default=None,
        help="Delay lights to match Bluetooth speakers (default ~200)",
    )
    music.add_argument(
        "--lead-ms",
        type=float,
        default=None,
        help="Fire lights early to cancel analysis/BLE lag (default ~50)",
    )
    music.add_argument("--preview", action="store_true", help="Print colors; skip BLE writes")

    sub.add_parser("devices", help="List audio capture devices")

    party = sub.add_parser("party", help="Music mode tuned for club / house / rap hype")
    party.add_argument("--address", default=None)
    party.add_argument("--source", choices=["system", "mic"], default=None)
    party.add_argument("--gain", type=float, default=None)
    party.add_argument(
        "--delay-ms",
        type=float,
        default=None,
        help="Delay lights to match Bluetooth speakers (default ~200)",
    )
    party.add_argument(
        "--lead-ms",
        type=float,
        default=None,
        help="Fire lights early to cancel analysis/BLE lag (default ~50)",
    )
    party.add_argument("--preview", action="store_true", help="Print colors; skip BLE writes")

    festival = sub.add_parser(
        "festival",
        help="High-energy festival mode: faster rainbow churn, bigger color jumps",
    )
    festival.add_argument("--address", default=None)
    festival.add_argument("--source", choices=["system", "mic"], default=None)
    festival.add_argument("--gain", type=float, default=None)
    festival.add_argument(
        "--delay-ms",
        type=float,
        default=None,
        help="Delay lights to match Bluetooth speakers (default ~200)",
    )
    festival.add_argument(
        "--lead-ms",
        type=float,
        default=None,
        help="Fire lights early to cancel analysis/BLE lag (default ~50)",
    )
    festival.add_argument("--preview", action="store_true", help="Print colors; skip BLE writes")

    party2 = sub.add_parser(
        "party2",
        help="v2 test: same structure as party, colors from sound association (timbre→hue)",
    )
    party2.add_argument("--address", default=None)
    party2.add_argument("--source", choices=["system", "mic"], default=None)
    party2.add_argument("--gain", type=float, default=None)
    party2.add_argument("--delay-ms", type=float, default=None)
    party2.add_argument("--lead-ms", type=float, default=None)
    party2.add_argument("--preview", action="store_true", help="Print colors; skip BLE writes")

    festival2 = sub.add_parser(
        "festival2",
        help="v2 test: festival energy + timbre→color association",
    )
    festival2.add_argument("--address", default=None)
    festival2.add_argument("--source", choices=["system", "mic"], default=None)
    festival2.add_argument("--gain", type=float, default=None)
    festival2.add_argument("--delay-ms", type=float, default=None)
    festival2.add_argument("--lead-ms", type=float, default=None)
    festival2.add_argument("--preview", action="store_true", help="Print colors; skip BLE writes")

    calm = sub.add_parser(
        "calm",
        help="Soft washes for chill / sad music (Mac DeMarco, Pink Floyd vibes)",
    )
    calm.add_argument("--address", default=None)
    calm.add_argument("--source", choices=["system", "mic"], default=None)
    calm.add_argument("--gain", type=float, default=None)
    calm.add_argument(
        "--delay-ms",
        type=float,
        default=None,
        help="Delay lights to match Bluetooth speakers (default ~200)",
    )
    calm.add_argument(
        "--lead-ms",
        type=float,
        default=None,
        help="Fire lights early to cancel analysis/BLE lag (default ~50)",
    )
    calm.add_argument("--preview", action="store_true", help="Print colors; skip BLE writes")

    darkside = sub.add_parser(
        "darkside",
        help="Pink Floyd Dark Side of the Moon: track-aware prism / heartbeat light show",
    )
    darkside.add_argument("--address", default=None)
    darkside.add_argument("--source", choices=["system", "mic"], default=None)
    darkside.add_argument("--gain", type=float, default=None)
    darkside.add_argument(
        "--delay-ms",
        type=float,
        default=None,
        help="Delay lights to match Bluetooth speakers (default ~200)",
    )
    darkside.add_argument(
        "--lead-ms",
        type=float,
        default=None,
        help="Fire lights early to cancel analysis/BLE lag (default ~50)",
    )
    darkside.add_argument("--preview", action="store_true", help="Print colors; skip BLE writes")

    cal = sub.add_parser(
        "calibrate-delay",
        help="Measure Bluetooth speaker lag vs loopback (uses mic)",
    )
    cal.add_argument("--clicks", type=int, default=6)
    cal.add_argument(
        "--save",
        action="store_true",
        help="Write measured delay_ms into config.toml",
    )

    vis = sub.add_parser(
        "calibrate",
        help="Visual delay tuner: beep + light flash (no mic)",
    )
    vis.add_argument("--address", default=None)
    vis.add_argument(
        "--delay-ms",
        type=float,
        default=None,
        help="Starting delay (default: config reactive.delay_ms)",
    )
    vis.add_argument(
        "--preview",
        action="store_true",
        help="Screen flash only; skip BLE",
    )

    phone = sub.add_parser(
        "calibrate-phone",
        help="Calibrate delay from your phone (same Wi-Fi web UI)",
    )
    phone.add_argument("--host", default="0.0.0.0", help="Bind address")
    phone.add_argument("--port", type=int, default=8765)
    phone.add_argument(
        "--delay-ms",
        type=float,
        default=None,
        help="Starting delay (default: config reactive.delay_ms)",
    )

    color_cal = sub.add_parser(
        "calibrate-color",
        help="Calibrate strip colors from your phone (same Wi-Fi web UI)",
    )
    color_cal.add_argument("--host", default="0.0.0.0", help="Bind address")
    color_cal.add_argument("--port", type=int, default=8766)
    color_cal.add_argument(
        "--preview",
        action="store_true",
        help="UI only; skip BLE writes",
    )

    movie = sub.add_parser(
        "movie",
        help="Match lights to on-screen colors (Ambilight / Hue Sync style)",
    )
    movie.add_argument("--address", default=None)
    movie.add_argument(
        "--intensity",
        choices=["subtle", "cinema", "moderate", "immersive", "high", "maximum", "extreme"],
        default=None,
        help="Room brightness feel (default: cinema). subtle < cinema < immersive < maximum",
    )
    movie.add_argument(
        "--monitor",
        type=int,
        default=None,
        help="mss monitor index (1 = primary). Use --list-monitors",
    )
    movie.add_argument(
        "--list-monitors",
        action="store_true",
        help="List monitors and exit",
    )
    movie.add_argument(
        "--saturation",
        type=float,
        default=None,
        help="Saturation gain override (e.g. 1.4)",
    )
    movie.add_argument(
        "--min-brightness",
        type=float,
        default=None,
        help="Floor so black scenes keep a soft glow (0–1)",
    )
    movie.add_argument(
        "--hz",
        type=float,
        default=None,
        help="Capture / update rate (default: config movie.update_hz)",
    )
    movie.add_argument("--preview", action="store_true", help="Print colors; skip BLE writes")

    return parser


def main(argv: list[str] | None = None) -> int:
    import sys

    if argv is None:
        argv = sys.argv[1:]

    commands = {
        "scan",
        "explore",
        "on",
        "off",
        "color",
        "brightness",
        "music",
        "devices",
        "party",
        "calm",
        "darkside",
        "festival",
        "party2",
        "festival2",
        "calibrate-delay",
        "calibrate",
        "calibrate-phone",
        "calibrate-color",
        "movie",
    }
    has_command = any(arg in commands for arg in argv)
    if not has_command:
        argv = [*argv, "party"]

    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    config = load_config(args.config)

    if args.command == "scan":
        return asyncio.run(print_scan(timeout=args.timeout, all_devices=args.all))
    if args.command == "devices":
        list_audio_devices()
        return 0
    if args.command == "calibrate":
        return _run_visual_calibrate(args, config)
    if args.command == "calibrate-phone":
        return _run_phone_calibrate(args, config)
    if args.command == "calibrate-color":
        return _run_phone_color_calibrate(args, config)
    if args.command == "calibrate-delay":
        return _run_calibrate_delay(args)
    if args.command == "explore":
        address = _require_address(args.address, config)
        return asyncio.run(explore_services(address))
    if args.command == "on":
        return asyncio.run(_simple_power(True, args.address, config))
    if args.command == "off":
        return asyncio.run(_simple_power(False, args.address, config))
    if args.command == "color":
        return asyncio.run(_set_color(args.r, args.g, args.b, args.address, config))
    if args.command == "brightness":
        return asyncio.run(_set_brightness(args.level, args.address, config))
    if args.command == "music":
        return asyncio.run(run_music_mode(config, args))
    if args.command == "party":
        args.mode = "party"
        return asyncio.run(run_music_mode(config, args))
    if args.command == "calm":
        args.mode = "calm"
        return asyncio.run(run_music_mode(config, args))
    if args.command == "darkside":
        args.mode = "darkside"
        return asyncio.run(run_music_mode(config, args))
    if args.command == "festival":
        args.mode = "festival"
        return asyncio.run(run_music_mode(config, args))
    if args.command == "party2":
        args.mode = "party2"
        return asyncio.run(run_music_mode(config, args))
    if args.command == "festival2":
        args.mode = "festival2"
        return asyncio.run(run_music_mode(config, args))
    if args.command == "movie":
        if args.list_monitors:
            return _list_monitors()
        return asyncio.run(run_movie_mode(config, args))
    parser.error(f"Unknown command: {args.command}")
    return 2


def _list_monitors() -> int:
    from rgb_lights.screen import list_monitors

    mons = list_monitors()
    if not mons:
        print("No monitors found.")
        return 1
    print("Monitors (use --monitor N):")
    for m in mons:
        print(f"  {m.label}")
    return 0


def _run_calibrate_delay(args: argparse.Namespace) -> int:
    from rgb_lights.calibrate import measure_speaker_delay, write_delay_to_config

    try:
        estimate = measure_speaker_delay(clicks=args.clicks)
    except Exception as exc:  # noqa: BLE001
        print(f"Calibration failed: {exc}")
        return 1

    samples = ", ".join(f"{x:.0f}" for x in estimate.samples)
    print(f"\nPer-click lags (ms): {samples}")
    print(f"Median delay: {estimate.delay_ms:.0f} ms")
    print("Use this as reactive.delay_ms so lights wait for the Bluetooth speaker.")
    if args.save:
        path = write_delay_to_config(estimate.delay_ms)
        print(f"Saved delay_ms = {int(round(estimate.delay_ms))} to {path}")
    else:
        print(f"Re-run with --save to write it, or: python -m rgb_lights party --delay-ms {int(round(estimate.delay_ms))}")
    return 0


def _run_visual_calibrate(args: argparse.Namespace, config: AppConfig) -> int:
    from rgb_lights.visual_calibrate import run_visual_calibrate

    try:
        result = run_visual_calibrate(
            config,
            address=args.address,
            preview=bool(args.preview),
            start_delay_ms=args.delay_ms,
            config_path=args.config or Path.cwd() / "config.toml",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Visual calibration failed: {exc}")
        return 1

    rounded = int(round(result.delay_ms))
    if result.saved and result.path is not None:
        print(f"Saved delay_ms = {rounded} to {result.path}")
    else:
        print(f"Current delay_ms = {rounded} (not saved)")
        print(f"Run party with: python -m rgb_lights party --delay-ms {rounded}")
        print("Or press S in the calibrator to write config.toml")
    return 0


def _run_phone_calibrate(args: argparse.Namespace, config: AppConfig) -> int:
    from rgb_lights.phone_calibrate import run_phone_calibrate

    try:
        return run_phone_calibrate(
            config,
            host=args.host,
            port=args.port,
            start_delay_ms=args.delay_ms,
            config_path=args.config or Path.cwd() / "config.toml",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Phone calibration failed: {exc}")
        return 1


def _run_phone_color_calibrate(args: argparse.Namespace, config: AppConfig) -> int:
    from rgb_lights.phone_color_calibrate import run_phone_color_calibrate

    try:
        return run_phone_color_calibrate(
            config,
            host=args.host,
            port=args.port,
            preview=bool(args.preview),
            config_path=args.config or Path.cwd() / "config.toml",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Phone color calibration failed: {exc}")
        return 1


def _require_address(cli_address: str | None, config: AppConfig) -> str:
    # CLI override > config.toml > hardcoded P031 defaults
    return cli_address or config.device.address


async def _with_controller(address: str | None, config: AppConfig) -> BleLightController:
    return BleLightController(
        address=_require_address(address, config),
        characteristic=config.device.characteristic or None,
    )


async def _simple_power(on: bool, address: str | None, config: AppConfig) -> int:
    controller = await _with_controller(address, config)
    async with controller:
        if on:
            await controller.power_on()
            print("Lights on")
        else:
            await controller.power_off()
            print("Lights off")
    return 0


async def _set_color(r: int, g: int, b: int, address: str | None, config: AppConfig) -> int:
    controller = await _with_controller(address, config)
    async with controller:
        await controller.power_on()
        await controller.set_rgb(r, g, b, force=True)
        print(f"Color set to rgb({r}, {g}, {b})")
    return 0


async def _set_brightness(level: int, address: str | None, config: AppConfig) -> int:
    controller = await _with_controller(address, config)
    async with controller:
        await controller.set_brightness(level)
        print(f"Brightness set to {level}")
    return 0


async def run_music_mode(config: AppConfig, args: argparse.Namespace) -> int:
    mode = args.mode or config.reactive.mode
    source = args.source or config.audio.source
    gain = args.gain if args.gain is not None else config.reactive.gain
    delay_ms = args.delay_ms if getattr(args, "delay_ms", None) is not None else config.reactive.delay_ms
    lead_ms = args.lead_ms if getattr(args, "lead_ms", None) is not None else config.reactive.lead_ms
    lead_ms = max(0.0, float(lead_ms))
    effective_delay_ms = max(0.0, float(delay_ms) - lead_ms)
    preview = bool(args.preview)
    use_director = mode in ("party", "calm", "darkside", "party2", "festival2")

    capture = AudioCapture(
        source=source,
        sample_rate=config.audio.sample_rate,
        chunk_size=config.audio.chunk_size,
    )
    pipeline: LightingPipeline | None = None
    analyzer: AudioAnalyzer | None = None
    mapper: ColorMapper | None = None

    if use_director:
        variant = "v1"
        if mode == "calm":
            variant = "calm"
        elif mode == "darkside":
            variant = "darkside"
        elif mode == "party2":
            variant = "party2"
        elif mode == "festival2":
            variant = "festival2"
        smoothing = config.reactive.smoothing
        if mode == "festival2":
            smoothing = min(smoothing, 0.22)
        elif mode == "calm":
            # Heavy band smoothing — glacial fades, no flicker
            smoothing = max(smoothing, 0.70)
        elif mode == "darkside":
            # Extra-smooth bands so colour melts between instruments / sections
            smoothing = max(smoothing, 0.55)
        pipeline = LightingPipeline(
            intensity=config.reactive.intensity,
            gain=gain,
            smoothing=smoothing,
            variant=variant,
        )
    else:
        # Festival needs snappier bands so hats/melody read through.
        smoothing = config.reactive.smoothing
        if mode == "festival":
            smoothing = min(smoothing, 0.22)
        analyzer = AudioAnalyzer(
            sample_rate=config.audio.sample_rate,
            gain=gain,
            smoothing=smoothing,
        )
        mapper = ColorMapper(mode=mode)

    limiter = RateLimiter(config.reactive.update_hz)
    if mode == "calm":
        # More intermediate frames so glacial RGB fades don't stair-step on the strip
        limiter = RateLimiter(max(config.reactive.update_hz, 20.0))
    delay = OutputDelay(effective_delay_ms)

    controller: BleLightController | None = None
    if not preview:
        controller = await _with_controller(args.address, config)
        await controller.connect()
        await controller.power_on()

    print(f"Music mode: {mode} | audio source: {source}")
    if mode == "calm":
        print(
            "Calm mode: sax/synth washes — glacial fades (no snaps), vivid color"
        )
    elif mode == "darkside":
        print(
            "Dark Side of the Moon: classifies Speak/Breathe/On the Run/Time/"
            "Great Gig/Money/Us and Them/Any Colour/Brain Damage/Eclipse — "
            "Hipgnosis ROYGBV prism (no indigo), heartbeat, Eclipse blooms"
        )
    elif mode == "party2":
        print(
            "party2: timbre→color (kick=warm, mids=green-gold, hats=blue) + beat/structure brightness"
        )
    elif mode == "festival2":
        print(
            "festival2: same color association as party2, snappier + bigger brightness swings"
        )
    elif use_director:
        print("Lighting director: buildup / drop / peak aware — beats locked to the groove")
    elif mode == "festival":
        print(
            "Festival mode: pace follows the song — mids=melody, highs=hats "
            "(same delay/lead/silence sync as party)"
        )
    print(
        f"Output delay: {delay_ms:.0f} ms - lead {lead_ms:.0f} ms "
        f"= {effective_delay_ms:.0f} ms effective | intensity: {config.reactive.intensity:.2f}"
    )
    if preview:
        print("Preview only - not writing to BLE")
    else:
        assert controller is not None
        print(f"Connected to {controller.address}")
        print("Listening to PC speakers (system audio loopback)")
        print("Tip: close the phone LED app — it can steal the BLE link and lights will die.")

    capture.start()
    await asyncio.sleep(0.15)
    if capture.device_label:
        print(f"Capturing via {capture.backend_name}: {capture.device_label}")
    print("Play music on this PC. Ctrl+C to stop.\n")

    silence_frames = 0
    was_silent = False
    idle_rgb = (0, 0, 0)
    ble_fail_streak = 0
    last_ble_ok = time.perf_counter()
    reconnecting = False

    try:
        while True:
            if controller is not None and not reconnecting and not controller.is_connected:
                reconnecting = True
                print("\nBLE disconnected — reconnecting...")
                try:
                    await controller.reconnect()
                    last_ble_ok = time.perf_counter()
                    ble_fail_streak = 0
                    print("BLE reconnected.")
                except Exception as exc:  # noqa: BLE001
                    print(f"BLE reconnect failed: {exc}")
                    await asyncio.sleep(0.5)
                finally:
                    reconnecting = False

            frame = await asyncio.to_thread(capture.read, 0.05)
            if frame is not None:
                # True pause only — quiet song sections must NOT kill the lights.
                samples = np.asarray(frame.samples, dtype=np.float32)
                rms = float(np.sqrt(np.mean(np.square(samples))))
                peak = float(np.max(np.abs(samples))) if samples.size else 0.0
                if rms < 0.0012 and peak < 0.004:
                    silence_frames += 1
                else:
                    if was_silent and controller is not None:
                        try:
                            await controller.power_on()
                            last_ble_ok = time.perf_counter()
                        except Exception:
                            pass
                    silence_frames = 0
                    was_silent = False

                # ~1.5s of near-digital-silence before treating as pause
                if silence_frames >= 28:
                    if not was_silent:
                        delay.clear()
                        was_silent = True
                        print(
                            f"\rRGB(  0,  0,  0)       {'silence':<18} bpm=  --   [----------------]",
                            end="",
                            flush=True,
                        )
                        if controller is not None:
                            try:
                                await controller.set_rgb(*idle_rgb, force=True)
                                last_ble_ok = time.perf_counter()
                            except Exception as exc:  # noqa: BLE001
                                print(f"\nBLE write failed: {exc}")
                                ble_fail_streak += 1
                    await asyncio.sleep(0.01)
                    continue

                important = False
                if use_director:
                    assert pipeline is not None
                    result = pipeline.process(frame.samples, frame.sample_rate)
                    if mode == "calm" and result.mood is not None:
                        mood = result.mood
                        # Soft updates only — never force on beats
                        if important or limiter.ready():
                            delay.push(
                                result.rgb,
                                beat=False,
                                downbeat=False,
                                bpm=0.0,
                                flux=mood.texture,
                                bass=mood.warmth,
                                energy=mood.voice if mood.voice > mood.level else mood.level,
                                state=mood.label,
                                palette=mood.palette,
                                tension=mood.open,
                                pre_drop=False,
                                drop_event=False,
                            )
                    else:
                        assert result.context is not None and result.intent is not None
                        important = (
                            result.context.beat
                            or result.context.drop_event
                            or result.context.pre_drop
                            or result.features.downbeat
                        )
                        if important or limiter.ready():
                            state_label = (
                                result.intent.label
                                if mode == "darkside"
                                else result.context.state.value
                            )
                            delay.push(
                                result.rgb,
                                beat=result.context.beat,
                                downbeat=result.context.downbeat,
                                bpm=result.context.bpm,
                                flux=result.features.flux,
                                bass=result.context.bass_energy,
                                energy=result.context.energy,
                                state=state_label,
                                palette=result.intent.palette.name,
                                tension=result.context.tension,
                                pre_drop=result.context.pre_drop,
                                drop_event=result.context.drop_event,
                            )
                else:
                    assert analyzer is not None and mapper is not None
                    features = analyzer.analyze(frame.samples, frame.sample_rate)
                    bands = SpectrumBands.from_features(features)
                    # Festival always maps so melody/hat state tracks every frame.
                    if mode == "festival":
                        rgb = mapper.map(bands)
                        important = bands.beat or mapper.urgent
                        if important or limiter.ready():
                            delay.push(
                                rgb,
                                beat=bands.beat,
                                bpm=bands.bpm,
                                flux=bands.flux,
                                bass=bands.bass,
                                energy=bands.energy,
                                state=mode,
                            )
                    else:
                        important = bands.beat
                        if important or limiter.ready():
                            rgb = mapper.map(bands)
                            delay.push(
                                rgb,
                                beat=bands.beat,
                                bpm=bands.bpm,
                                flux=bands.flux,
                                bass=bands.bass,
                                energy=bands.energy,
                                state=mode,
                            )

            ready = delay.pop_all_ready()
            if not ready:
                await asyncio.sleep(0.005)
                continue

            out = ready[-1]
            for item in reversed(ready):
                if item.beat or item.downbeat:
                    out = item
                    break

            bar = _level_bar(out.energy)
            r, g, b = out.rgb
            if mode == "calm":
                mark = {
                    "space": " space",
                    "swell": " swell",
                    "bloom": " bloom",
                    "glow": "  glow",
                    "haze": "  haze",
                }.get(out.state, "     ")
                state_txt = f"{out.state}/{out.palette}" if out.palette else out.state
                meters = f" w={out.bass:.2f} v={out.energy:.2f} o={out.tension:.2f}"
                print(
                    f"\rRGB({r:3d},{g:3d},{b:3d}){mark}  "
                    f"{state_txt:<18}{meters}  {bar}",
                    end="",
                    flush=True,
                )
            else:
                if out.drop_event or out.state == "drop":
                    mark = " DROP"
                elif out.pre_drop:
                    mark = " hush"
                elif out.downbeat:
                    mark = "  DB "
                elif out.beat:
                    mark = " beat"
                else:
                    mark = "     "
                bpm_txt = f"{out.bpm:5.1f}" if out.bpm else "  -- "
                state_txt = f"{out.state}/{out.palette}" if out.palette else out.state
                tens = f" t={out.tension:.2f}" if out.tension > 0.2 else ""
                print(
                    f"\rRGB({r:3d},{g:3d},{b:3d}){mark}  "
                    f"{state_txt:<18} bpm={bpm_txt}{tens}  {bar}",
                    end="",
                    flush=True,
                )
            if controller is not None:
                try:
                    force = False if mode == "calm" else (out.beat or out.downbeat)
                    await controller.set_rgb(r, g, b, force=force)
                    last_ble_ok = time.perf_counter()
                    ble_fail_streak = 0
                except Exception as exc:  # noqa: BLE001
                    ble_fail_streak += 1
                    print(f"\nBLE write failed: {exc}")
                    if ble_fail_streak >= 1:
                        print("Reconnecting BLE...")
                        try:
                            await controller.reconnect()
                            last_ble_ok = time.perf_counter()
                            ble_fail_streak = 0
                            await controller.set_rgb(r, g, b, force=True)
                            print("BLE reconnected.")
                        except Exception as exc2:  # noqa: BLE001
                            print(f"BLE reconnect failed: {exc2}")
                            await asyncio.sleep(0.4)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        capture.stop()
        if controller is not None:
            try:
                await controller.set_rgb(0, 0, 0, force=True)
                await controller.power_off()
            except Exception:
                pass
            await controller.disconnect()
    return 0


async def run_movie_mode(config: AppConfig, args: argparse.Namespace) -> int:
    """Screen-edge ambient sync — Hue Sync Movie / Hyperion style."""
    from rgb_lights.lighting.movie import MovieEngine
    from rgb_lights.reactive import RateLimiter
    from rgb_lights.screen import ScreenCapture, ScreenSampler

    mc = config.movie
    intensity = args.intensity or mc.intensity
    monitor = args.monitor if args.monitor is not None else mc.monitor
    sat = args.saturation if args.saturation is not None else (mc.saturation or None)
    min_b = args.min_brightness if args.min_brightness is not None else mc.min_brightness
    hz = args.hz if args.hz is not None else mc.update_hz
    preview = bool(args.preview)
    smooth = mc.smoothing if mc.smoothing > 0 else None

    try:
        capture = ScreenCapture(monitor=monitor, max_width=96)
    except Exception as exc:  # noqa: BLE001
        print(f"Screen capture failed: {exc}")
        print("Tip: python -m rgb_lights movie --list-monitors")
        return 1

    sampler = ScreenSampler(
        edge_depth=mc.edge_depth,
        blackbar=mc.blackbar,
        blackbar_mode=mc.blackbar_mode,
        highlight_mix=getattr(mc, "highlight_mix", 0.25),
        black_luma=getattr(mc, "black_luma", 0.028),
    )
    engine = MovieEngine(
        intensity=intensity,
        saturation=sat,
        min_brightness=min_b,
        global_intensity=config.reactive.intensity,
        smoothing=smooth,
        r_gain=mc.r_gain,
        g_gain=mc.g_gain,
        b_gain=mc.b_gain,
        color_matrix=mc.color_matrix,
        black_luma=getattr(mc, "black_luma", 0.028),
    )
    limiter = RateLimiter(hz)

    controller: BleLightController | None = None
    if not preview:
        controller = await _with_controller(args.address, config)
        try:
            await controller.connect()
            await controller.power_on()
        except Exception as exc:  # noqa: BLE001
            print(f"\nCould not connect to lights:\n{exc}")
            return 1

    print(f"Movie mode: {engine.intensity_name} | {capture.label}")
    print(
        f"Ambient model: edge-weighted linear light | room_scale={engine.room_scale:.2f} | "
        f"black_luma={getattr(mc, 'black_luma', 0.028):.3f} | {hz:.0f} Hz"
    )
    print(
        f"Temporal: attack={engine.attack_s*1000:.0f}ms  release={engine.release_s*1000:.0f}ms  "
        f"color={engine.color_s*1000:.0f}ms"
    )
    if preview:
        print("Preview only - not writing to BLE")
    else:
        assert controller is not None
        print(f"Connected to {controller.address}")
        print("Tip: close the phone LED app — it can steal the BLE link.")
    print("Watch something fullscreen. Ctrl+C to stop.\n")

    ble_fail_streak = 0
    reconnecting = False

    try:
        while True:
            if controller is not None and not reconnecting and not controller.is_connected:
                reconnecting = True
                print("\nBLE disconnected — reconnecting...")
                try:
                    await controller.reconnect()
                    ble_fail_streak = 0
                    print("BLE reconnected.")
                except Exception as exc:  # noqa: BLE001
                    print(f"BLE reconnect failed: {exc}")
                    await asyncio.sleep(0.5)
                finally:
                    reconnecting = False

            if not limiter.ready():
                await asyncio.sleep(0.005)
                continue

            try:
                frame = await asyncio.to_thread(capture.grab)
            except Exception as exc:  # noqa: BLE001
                print(f"\nCapture error: {exc}")
                await asyncio.sleep(0.25)
                continue

            sample = sampler.sample(frame)
            result = engine.process(sample)
            r, g, b = result.rgb
            bar = _level_bar(result.luma)
            mark = " CUT" if getattr(result, "cut", False) else (" blk" if sample.black else "    ")
            print(
                f"\rRGB({r:3d},{g:3d},{b:3d}){mark}  "
                f"{result.intensity:<9} L={result.luma:.2f}  {bar}   ",
                end="",
                flush=True,
            )

            if controller is not None:
                try:
                    await controller.set_rgb(r, g, b)
                    ble_fail_streak = 0
                except Exception as exc:  # noqa: BLE001
                    ble_fail_streak += 1
                    print(f"\nBLE write failed: {exc}")
                    if ble_fail_streak >= 1:
                        print("Reconnecting BLE...")
                        try:
                            await controller.reconnect()
                            ble_fail_streak = 0
                            await controller.set_rgb(r, g, b, force=True)
                            print("BLE reconnected.")
                        except Exception as exc2:  # noqa: BLE001
                            print(f"BLE reconnect failed: {exc2}")
                            await asyncio.sleep(0.4)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        capture.close()
        if controller is not None:
            try:
                await controller.set_rgb(0, 0, 0, force=True)
                await controller.power_off()
            except Exception:
                pass
            await controller.disconnect()
    return 0


def _level_bar(energy: float, width: int = 16) -> str:
    filled = int(round(_clamp01(energy) * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
