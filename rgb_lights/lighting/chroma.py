from __future__ import annotations

"""Map sound character (timbre) to color so similar noises look similar.

Rough chromesthesia-style layout (stays consistent over time):
  sub / kick  → deep red–orange
  warm bass   → amber
  body / mids → yellow–green (melody, vocals)
  presence    → cyan
  air / hats  → blue–violet
  noisy flux  → magenta pull
"""


def timbre_to_hsv(
    sub: float,
    bass: float,
    mid: float,
    high: float,
    flux: float = 0.0,
    onset_strength: float = 0.0,
) -> tuple[float, float, float]:
    """Return (hue, sat, brightness_hint) in 0–1 from band mix."""
    sub = _c(sub)
    bass = _c(bass)
    mid = _c(mid)
    high = _c(high)
    flux = _c(flux)
    onset = _c(onset_strength)

    total = sub + bass + mid + high + 1e-6
    w_sub = sub / total
    w_bass = bass / total
    w_mid = mid / total
    w_high = high / total

    # Spectral centroid proxy (dark → bright timbre)
    centroid = w_sub * 0.05 + w_bass * 0.18 + w_mid * 0.48 + w_high * 0.85

    # Wider hue map so band shifts read clearly on a strip
    # red → orange → gold → green → cyan → blue → violet → magenta
    hue = (
        0.00 * w_sub
        + 0.08 * w_bass
        + 0.36 * w_mid
        + 0.68 * w_high
    )
    # Blend with centroid for extra spread across the wheel
    hue = 0.45 * hue + 0.55 * (centroid * 0.92)
    # Transients / noise pull toward magenta-pink (distinct from kicks)
    hue = (hue + flux * 0.18 + onset * 0.08) % 1.0

    sat = 0.88 + w_bass * 0.10 + w_mid * 0.05 - w_high * 0.06
    sat = _c(min(0.98, sat))

    # How "loud" this timbre feels for brightness bias
    hint = _c(0.55 * bass + 0.25 * mid + 0.12 * high + 0.08 * sub + 0.15 * onset)
    return hue % 1.0, sat, hint


def _c(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
