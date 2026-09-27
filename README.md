# RGB Lights (Daybetter)

Control Daybetter Bluetooth LED strips from Python with [Bleak](https://github.com/hbldh/bleak), including music-reactive mode for audio playing on your PC.

## Setup

```powershell
cd "C:\Users\gavin\Documents\Coding\RGB Lights"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
copy config.example.toml config.toml
```

## Connect your Daybetter strip

1. Force-close the phone app (only one BLE client can connect).
2. Power on the strip.
3. Scan:

```powershell
python -m rgb_lights scan
# wider BLE scan if needed:
python -m rgb_lights scan --all
```

4. Put the address in `config.toml`:

```toml
[device]
address = "AA:BB:CC:DD:EE:FF"
```

5. Test:

```powershell
python -m rgb_lights on
python -m rgb_lights color 255 40 0
python -m rgb_lights brightness 80
python -m rgb_lights off
```

If color fails after connect, inspect characteristics:

```powershell
python -m rgb_lights explore
```

Then set `characteristic` in `config.toml` if needed.

## Music / party mode

Party mode uses a **Lighting Director** (not flash-on-every-beat):

```text
Audio → Analyzer → Music state → Director → Effects → BLE (+ BT delay)
```

- Subtle beat *breathing* (~5–15%), stronger on downbeats
- Curated palettes held for ~15–60s (`deep`, `club`, `warm`, `dreamy`, `hype`, `chill`)
- Buildup / drop / breakdown states with rare accents (warm punch, not white strobe)
- `delay_ms` syncs to Bluetooth speakers; `intensity` caps dorm brightness

```powershell
python -m rgb_lights party
python -m rgb_lights party --delay-ms 250
```

Calm mode is for chill / sad music (Mac DeMarco, Pink Floyd vibes). It ignores
BPM and beats — it follows **sax/synth** body (250–900 Hz) and lead (900–3200 Hz)
bands, plus **warmth**, **space**, and **swell**, with slew-limited color melts:

```powershell
python -m rgb_lights calm
python -m rgb_lights music --mode calm
```

**Dark Side of the Moon** mode is built for that album — live audio is classified
into track-like movements (not Spotify IDs) and lit like the Hipgnosis cover +
1974 show:

| Movement | Feel |
|----------|------|
| Speak / Heartbeat | Void + lub-dub pulse, foreshadow colour flickers |
| Breathe | Warm floating wash, slow Leslie drift |
| On the Run | Anxious violet scramble, alarm flashes |
| Time | Clock-tick gold cascade → rock spectrum |
| Great Gig | Mid/vocal drama, rose↔violet swells |
| Money | Cash-gold sparkle + rubber-bass punch |
| Us and Them | Soft blue void ↔ loud prism chorus breath |
| Any Colour | Continuous ROYGBV spectrum crawl |
| Brain Damage | Eerie green-violet, laughter ticks |
| Eclipse | White recombine → full fan bloom |

Spectrum matches the cover (**indigo omitted**). Status line shows the active movement.

```powershell
python -m rgb_lights darkside
.\run_darkside.bat
```

Festival mode is separate — faster rainbow churn, bigger beat color jumps, neon punch:

```powershell
python -m rgb_lights festival
python -m rgb_lights music --mode festival
```

Legacy modes: `music --mode spectrum|bass_pulse|energy`

## Movie mode (screen sync)

Ambient wall lighting for movies — not a gaming RGB sampler. Estimates what
light the scene should cast into the room (edge-weighted linear light, separate
color vs brightness, highlight pull, true blacks, fast flash / slow fade).

```powershell
python -m rgb_lights movie
python -m rgb_lights movie --intensity cinema
python -m rgb_lights movie --intensity subtle
python -m rgb_lights movie --intensity immersive
python -m rgb_lights movie --list-monitors
python -m rgb_lights movie --preview
```

| Preset | Feel |
|--------|------|
| `subtle` | Soft room wash — won't fight the screen |
| `cinema` | Default dorm balance |
| `immersive` | Stronger walls, still controlled |
| `maximum` | Brightest calibrated room scale |

`reactive.intensity` still caps overall room scale. Phone `calibrate-color`
fits a color matrix so LED+wall response matches what you want.

### Phone color calibration

Match the strip to your phone camera over Wi‑Fi (HTTPS). Auto-tune samples
**white + RGB + cyan/magenta/yellow/orange** and fits a **3×3 color matrix**
(full gamut — not just R/G/B sliders). Built for iPhone Safari.

```powershell
python -m rgb_lights calibrate-color
# or:
.\run_calibrate_color.bat
```

Open the printed **https://** URL on your iPhone (accept the certificate warning),
allow camera, fill the yellow box with the strip, tap **Auto tune**, then **Save**.

Close the calibrator before running movie mode (only one BLE connection).

```powershell
python -m rgb_lights devices
```

## Config knobs

- `reactive.update_hz` — keep around 10-20
- `reactive.smoothing` — higher = smoother
- `reactive.gain` — raise if colors stay dim
- `audio.source` — `system` (PC playback) or `mic`
- `movie.intensity` — `subtle` / `cinema` / `immersive` / `maximum`
- `movie.monitor` — which display to follow (`movie --list-monitors`)
- `movie.black_luma` — below this scene luminance, strip fades off
- `movie.highlight_mix` — how much explosions/sun pull the color
- `movie.edge_depth` — outer-frame weight for room bleed (~0.32)
- `movie.blackbar` — crop letterboxing before sampling
- `audio.source` — `system` (PC playback) or `mic`
