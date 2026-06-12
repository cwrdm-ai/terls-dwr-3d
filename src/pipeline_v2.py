"""Pipeline v2 — the MOSDAC-documented processing chain.

Differences from pipeline.py (v1):
  - Clutter QC: Gabella 2002 spatial continuity + Vulpiani 2012 fuzzy
    classification (clutter_correct_v2) instead of our RHOHV gate
  - Gridding: tight constant radius-of-influence (~700 m) approximating the
    wradlib interpolation MOSDAC used, instead of beam-spreading Barnes

Outputs *_gridded_v2.nc so v1 and v2 cubes can be compared side by side.

Usage:
    python src/pipeline_v2.py data/Jun26_182954/RCTLS_06JUN2026_051420_L2B_STD.nc
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import io_radar
import clutter_correct_v2
import dealias_velocity
import grid_to_cartesian
import render_dwr


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"


def run(polar_nc: Path) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    stem = polar_nc.stem
    gridded_nc = OUTPUT_DIR / f"{stem}_gridded_v2.nc"

    t0 = time.time()

    print(f"\n[1/4] Reading polar volume from {polar_nc.name}...")
    radar = io_radar.read(polar_nc)
    print(f"      {radar.nsweeps} sweeps, {radar.nrays} rays, {radar.ngates} gates")

    print(f"\n[2/4] QC v2: Gabella spatial filter + Vulpiani fuzzy classification...")
    radar = clutter_correct_v2.correct(radar)

    print(f"\n[3/4] Velocity dealiasing...")
    radar = dealias_velocity.dealias(radar)

    print(f"\n[4/4] Gridding (tight 700 m RoI)...")
    grid_to_cartesian.grid(radar, gridded_nc, roi_func="constant", constant_roi=700.0)

    print(f"\nDone in {time.time() - t0:.1f}s -> {gridded_nc}")
    return gridded_nc


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python src/pipeline_v2.py <path-to-polar-nc>")
        sys.exit(1)
    run(Path(sys.argv[1]))
