from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Palette:
    name: str
    primary: tuple[float, float, float]  # HSV 0-1
    secondary: tuple[float, float, float]
    accent: tuple[float, float, float]


# Wide hue spreads so a single strip can swing across the spectrum.
PALETTES: dict[str, Palette] = {
    "deep": Palette(
        "deep",
        primary=(0.58, 0.95, 0.88),  # blue
        secondary=(0.78, 0.92, 0.90),  # violet
        accent=(0.95, 0.95, 0.92),  # magenta-pink
    ),
    "club": Palette(
        "club",
        primary=(0.52, 0.98, 0.92),  # cyan
        secondary=(0.78, 0.95, 0.92),  # purple
        accent=(0.00, 0.95, 0.95),  # red
    ),
    "warm": Palette(
        "warm",
        primary=(0.08, 0.98, 0.95),  # orange
        secondary=(0.00, 0.95, 0.92),  # red
        accent=(0.14, 0.95, 0.95),  # gold
    ),
    "dreamy": Palette(
        "dreamy",
        primary=(0.48, 0.90, 0.92),  # cyan
        secondary=(0.78, 0.70, 0.92),  # lavender
        accent=(0.92, 0.85, 0.92),  # pink
    ),
    "hype": Palette(
        "hype",
        primary=(0.00, 0.98, 0.98),  # red
        secondary=(0.88, 0.98, 0.98),  # magenta
        accent=(0.16, 0.98, 0.98),  # yellow
    ),
    "chill": Palette(
        "chill",
        primary=(0.42, 0.88, 0.88),  # teal
        secondary=(0.62, 0.90, 0.90),  # blue
        accent=(0.28, 0.92, 0.90),  # green
    ),
    "neon": Palette(
        "neon",
        primary=(0.30, 0.98, 0.98),  # lime
        secondary=(0.55, 0.98, 0.98),  # aqua
        accent=(0.92, 0.98, 0.98),  # hot pink
    ),
    "sunset": Palette(
        "sunset",
        primary=(0.95, 0.95, 0.95),  # rose
        secondary=(0.08, 0.98, 0.95),  # orange
        accent=(0.72, 0.92, 0.92),  # purple
    ),
    "electric": Palette(
        "electric",
        primary=(0.58, 0.98, 0.98),  # electric blue
        secondary=(0.14, 0.98, 0.98),  # yellow
        accent=(0.92, 0.98, 0.98),  # magenta
    ),
    # Calm / sad-music — near-max sat so strips don't wash to white
    "amber": Palette(
        "amber",
        primary=(0.09, 1.00, 0.92),
        secondary=(0.04, 1.00, 0.90),
        accent=(0.14, 0.98, 0.94),
    ),
    "dusk": Palette(
        "dusk",
        primary=(0.95, 1.00, 0.92),
        secondary=(0.84, 0.98, 0.90),
        accent=(0.06, 1.00, 0.92),
    ),
    "indigo": Palette(
        "indigo",
        primary=(0.66, 1.00, 0.92),
        secondary=(0.58, 0.98, 0.90),
        accent=(0.76, 0.98, 0.92),
    ),
    "fog": Palette(
        "fog",
        primary=(0.48, 0.95, 0.88),
        secondary=(0.55, 0.95, 0.88),
        accent=(0.40, 0.95, 0.88),
    ),
    "velvet": Palette(
        "velvet",
        primary=(0.84, 1.00, 0.92),
        secondary=(0.74, 0.98, 0.90),
        accent=(0.94, 1.00, 0.92),
    ),
    # Dark Side of the Moon — Hipgnosis prism / void
    "void": Palette(
        "void",
        primary=(0.62, 0.55, 0.55),  # deep space blue
        secondary=(0.72, 0.40, 0.48),  # faint violet
        accent=(0.58, 0.12, 0.70),  # ghost white-cyan beam
    ),
    "beam": Palette(
        "beam",
        primary=(0.58, 0.10, 0.92),  # cool white beam
        secondary=(0.55, 0.25, 0.88),  # pale ice
        accent=(0.00, 0.75, 0.90),  # prism red tip
    ),
    "prism": Palette(
        "prism",
        primary=(0.00, 0.96, 0.95),  # spectrum red
        secondary=(0.58, 0.92, 0.95),  # spectrum blue
        accent=(0.145, 0.94, 0.95),  # spectrum yellow
    ),
    "eclipse": Palette(
        "eclipse",
        primary=(0.78, 0.88, 0.94),  # violet
        secondary=(0.00, 0.95, 0.95),  # red
        accent=(0.58, 0.08, 0.96),  # recombined white
    ),
}


STATE_PALETTE_PREFERENCE = {
    "chill": ("chill", "dreamy", "neon", "deep"),
    "groove": ("club", "neon", "electric", "deep"),
    "energetic": ("electric", "hype", "neon", "sunset"),
    "buildup": ("deep", "club", "electric", "hype"),
    "drop": ("hype", "sunset", "electric", "warm"),
    "breakdown": ("dreamy", "chill", "sunset", "deep"),
    "peak": ("hype", "neon", "electric", "warm"),
}

# Mac DeMarco / Pink Floyd / sad-music — slow palette holds, no neon
CALM_STATE_PALETTE_PREFERENCE = {
    "chill": ("amber", "dusk", "fog", "indigo"),
    "groove": ("dusk", "amber", "velvet", "fog"),
    "energetic": ("indigo", "velvet", "dusk", "amber"),
    "buildup": ("fog", "indigo", "velvet", "dusk"),
    "drop": ("velvet", "dusk", "amber", "indigo"),
    "breakdown": ("indigo", "fog", "dusk", "amber"),
    "peak": ("dusk", "velvet", "amber", "indigo"),
}

# Dark Side of the Moon — void → beam → prism → eclipse
DARKSIDE_STATE_PALETTE_PREFERENCE = {
    "chill": ("void", "beam", "void", "prism"),
    "groove": ("beam", "prism", "void", "eclipse"),
    "energetic": ("prism", "eclipse", "beam", "void"),
    "buildup": ("void", "beam", "prism", "eclipse"),
    "drop": ("eclipse", "prism", "beam", "void"),
    "breakdown": ("void", "beam", "void", "prism"),
    "peak": ("prism", "eclipse", "prism", "beam"),
}


def get_palette(name: str) -> Palette:
    return PALETTES.get(name, PALETTES["club"])
