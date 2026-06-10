"""dBZ -> color + opacity transfer function for 3D DWR volume rendering.

This is the single most consequential design decision for how a storm looks.
Two channels:

  Color    : what hue is mapped to each reflectivity value.
             The meteorology convention is roughly green->yellow->red->magenta,
             mirroring NWS / IMD radar products. Forecasters read colors as a
             quick proxy for "is this drizzle or hail?".

  Opacity  : how see-through each value is. This is the secret sauce of 3D
             volume rendering. If opacity ramps up at 20 dBZ, you'll see every
             cloud — the scene becomes a foggy blob. If it ramps at 35+, only
             the convective cores show — clean storm anatomy.

==========================================================================
                       HOW TO MAKE THIS YOUR OWN
==========================================================================

Two scales are defined below: MOSDAC_STYLE (conservative — only storm cores
visible, matches MOSDAC's hero shots) and EDUCATIONAL_STYLE (aggressive — see
every cloud, foggy but complete). The pipeline uses ACTIVE_SCALE.

To change the look:
  - Easiest: change the line  ACTIVE_SCALE = MOSDAC_STYLE  to use
    EDUCATIONAL_STYLE, or write your own list.
  - Deeper: edit any breakpoint row — each is (dBZ, R, G, B, alpha) in [0, 1].

NWS reference scale (industry standard you can mimic or override):
   dBZ   | meaning            | typical color
   ------|--------------------|---------------------
     5   | very light drizzle | pale cyan / blue
    20   | light rain         | green
    35   | moderate rain      | yellow
    45   | heavy rain         | orange / red
    55   | very heavy / hail  | magenta
    65+  | large hail core    | white / pink

Trade-off:
  Aggressive opacity (everything visible) = great educational view but foggy.
  Conservative opacity (only cores) = clean storm anatomy. MOSDAC uses this.
"""
from __future__ import annotations

import numpy as np
from matplotlib.colors import LinearSegmentedColormap


# Conservative — matches MOSDAC's published 3D renders. Storm cores stand out
# against black; light precipitation is nearly invisible. Best for severe-
# weather aesthetics.
MOSDAC_STYLE: list[tuple[float, float, float, float, float]] = [
    # (dBZ,   R,    G,    B,    alpha)
    (  0.0, 0.00, 0.00, 0.30, 0.00),  # below noise: invisible
    ( 10.0, 0.00, 0.50, 0.80, 0.00),  # very light: invisible
    ( 20.0, 0.00, 0.80, 0.40, 0.02),  # light rain: green, barely there
    ( 30.0, 0.40, 1.00, 0.00, 0.10),  # moderate light: yellow-green
    ( 40.0, 1.00, 0.90, 0.00, 0.35),  # moderate-heavy: yellow, opaque-ish
    ( 50.0, 1.00, 0.30, 0.00, 0.65),  # heavy rain: orange-red
    ( 60.0, 1.00, 0.00, 0.50, 0.85),  # hail / very heavy: magenta
    ( 70.0, 1.00, 0.80, 1.00, 0.95),  # extreme hail core: pink-white
]


# Aggressive — see ALL the weather. Good for teaching what a storm looks like
# top to bottom. Sacrifices clean look for completeness.
EDUCATIONAL_STYLE: list[tuple[float, float, float, float, float]] = [
    (  0.0, 0.20, 0.20, 0.50, 0.00),
    (  5.0, 0.00, 0.30, 0.80, 0.05),   # drizzle: blue, visible
    ( 15.0, 0.00, 0.80, 0.80, 0.15),   # light rain: cyan
    ( 25.0, 0.00, 0.90, 0.20, 0.30),   # moderate: green
    ( 35.0, 1.00, 1.00, 0.00, 0.45),   # heavier: yellow
    ( 45.0, 1.00, 0.40, 0.00, 0.65),   # heavy: orange
    ( 55.0, 1.00, 0.00, 0.30, 0.80),   # very heavy: red-magenta
    ( 65.0, 1.00, 0.70, 1.00, 0.92),   # hail: pink
    ( 75.0, 1.00, 1.00, 1.00, 0.98),   # extreme: white
]


# ============================================================
#   >>> CHOOSE YOUR LOOK HERE <<<
# Set this to MOSDAC_STYLE, EDUCATIONAL_STYLE, or your own list.
#
# Switched to EDUCATIONAL_STYLE because this scan (09-Jun-2026 05:01 UTC) is
# monsoon stratiform precipitation: 88% of returns are in [-10, +30] dBZ.
# MOSDAC_STYLE (built for severe-weather cores) would render it invisible.
# For thunderstorm scans, switch back to MOSDAC_STYLE.
# ============================================================
ACTIVE_SCALE = EDUCATIONAL_STYLE
# ============================================================


def build_dbz_transfer_function():
    """Convert ACTIVE_SCALE into (matplotlib colormap, pyvista opacity list).

    PyVista wants opacity as a SHORT list whose entries match the colormap
    sampling — we hand it the raw breakpoint alphas (one per stop). PyVista
    interpolates linearly between them across clim, which is what we want.

    Colormap goes through the same logic: short list of (position, RGB).
    """
    if not ACTIVE_SCALE:
        raise ValueError("ACTIVE_SCALE is empty — set it to MOSDAC_STYLE or your own scale.")

    stops = sorted(ACTIVE_SCALE, key=lambda r: r[0])
    dbz_min, dbz_max = 0.0, 70.0

    # Position each stop along the clim range. Anything outside (e.g. 75 dBZ
    # when clim ends at 70) is clamped, and we then force the endpoints to
    # exactly 0.0 / 1.0 as matplotlib requires.
    raw_pos = [(d - dbz_min) / (dbz_max - dbz_min) for (d, *_rgba) in stops]
    norm_positions = [max(0.0, min(1.0, p)) for p in raw_pos]
    norm_positions[0] = 0.0
    norm_positions[-1] = 1.0

    colors = [(r, g, b) for (_d, r, g, b, _a) in stops]
    cmap = LinearSegmentedColormap.from_list(
        "dwr_dbz",
        list(zip(norm_positions, colors)),
        N=256,
    )

    # Opacity: hand PyVista the alpha values one per stop, IN ORDER.
    # PyVista linearly interpolates across clim — exactly what we want.
    opacity = [float(a) for (*_d, a) in stops]

    return cmap, opacity
