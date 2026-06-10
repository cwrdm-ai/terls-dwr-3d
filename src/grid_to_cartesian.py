"""Polar -> Cartesian gridding. This is the step that produces THE cube.

The MOSDAC paper specifies the output grid:
   horizontal: 1 km x 1 km, 481 x 481 cells = 480 km x 480 km
   vertical  : 250 m,        81 levels       = 0 to 20 km
   centered on the radar (TERLS at 8.54N, 76.87E)

pyart.map.grid_from_radars handles the heavy interpolation: for each Cartesian
cell, it gathers nearby polar gates within a radius-of-influence (RoI) and
weights them. Two interpolation choices matter:

  - **weighting_function**: 'Barnes2' is the default, an exponentially weighted
    average — smooth, used by NEXRAD MRMS. 'Cressman' is sharper but boxier.
  - **roi_func**: how far to search for polar gates per Cartesian cell.
    'dist_beam' scales with range (far gates spread wider so we need a wider
    search). This is the physically correct choice for cone-shaped radar beams.

Output: a pyart.Grid that we save to NetCDF in the MOSDAC layout
(altitude, latitude, longitude) — so the renderer can be source-agnostic.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyart


# ============================================================
#   Grid spec — match MOSDAC exactly.
# ============================================================
N_ALTITUDE = 81           # 0 -> 20 km, 250 m spacing
N_LAT = 481               # 480 km N-S, 1 km spacing
N_LON = 481               # 480 km E-W, 1 km spacing
ALT_MAX_M = 20_000.0
HALF_HORIZ_M = 240_000.0  # ±240 km from radar
# ============================================================


def grid(radar: pyart.core.Radar, output_nc: Path) -> Path:
    """Run gridding and write the resulting Cartesian cube to NetCDF."""
    print(
        f"Gridding to {N_ALTITUDE} x {N_LAT} x {N_LON} cube "
        f"(1 km x 1 km horiz, 250 m vert)..."
    )

    grid_obj = pyart.map.grid_from_radars(
        (radar,),
        grid_shape=(N_ALTITUDE, N_LAT, N_LON),
        grid_limits=(
            (0.0, ALT_MAX_M),
            (-HALF_HORIZ_M, HALF_HORIZ_M),
            (-HALF_HORIZ_M, HALF_HORIZ_M),
        ),
        fields=["DBZ", "VEL_DEALIASED"] if "VEL_DEALIASED" in radar.fields else ["DBZ", "VEL"],
        weighting_function="Barnes2",
        roi_func="dist_beam",
        h_factor=1.0,
        nb=1.5,
        bsp=1.0,
        min_radius=500.0,
    )

    output_nc.parent.mkdir(parents=True, exist_ok=True)
    pyart.io.write_grid(str(output_nc), grid_obj)

    dbz = grid_obj.fields["DBZ"]["data"]
    valid = dbz.compressed() if hasattr(dbz, "compressed") else dbz[np.isfinite(dbz)]
    if valid.size:
        print(
            f"Gridded DBZ: shape={dbz.shape}  "
            f"range=[{float(valid.min()):.1f}, {float(valid.max()):.1f}] dBZ  "
            f"valid cells={valid.size:,} / {dbz.size:,} ({100*valid.size/dbz.size:.1f}%)"
        )
    print(f"Saved gridded cube: {output_nc}")
    return output_nc
