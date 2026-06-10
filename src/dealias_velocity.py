"""Velocity dealiasing — unfold radial velocity that wraps past the Nyquist.

The radar can only unambiguously measure velocities between -Nyquist and
+Nyquist. For TERLS that's roughly ±24 m/s. Anything faster *appears* to wrap
around: a real +30 m/s shows up as -18 m/s. In a mesocyclone or strong jet
streak this is constant — and on a 3D viz those wrapped values would look like
inflow where there's actually outflow.

The MOSDAC paper specifies "2D multipass velocity dealiasing." pyart's
`dealias_region_based` algorithm is the closest open-source equivalent: it
identifies connected regions of similar velocity and unfolds them coherently,
iterating until stable. Same family of algorithm.

Reference: Bergen & Albers 1988; James & Houze 2001; Helmus & Collis 2016
(pyart paper).
"""
from __future__ import annotations

import warnings

import pyart


def dealias(radar: pyart.core.Radar) -> pyart.core.Radar:
    """Add a 'velocity_dealiased' field to the radar object."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dealiased = pyart.correct.dealias_region_based(
            radar,
            vel_field="VEL",
            keep_original=False,
            centered=True,
        )

    radar.add_field("VEL_DEALIASED", dealiased, replace_existing=True)

    vd = dealiased["data"]
    valid = vd.compressed() if hasattr(vd, "compressed") else vd[~vd.mask] if hasattr(vd, "mask") else vd
    if valid.size:
        print(
            f"Velocity dealiasing: VEL_DEALIASED range "
            f"[{float(valid.min()):.1f}, {float(valid.max()):.1f}] m/s "
            f"(original was clipped at ±24 m/s)"
        )

    return radar
