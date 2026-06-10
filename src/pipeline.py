"""End-to-end pipeline: MOSDAC polar .nc -> 3D volume render.

Steps (matches the MOSDAC 3D Volumetric TERLS DWRproduct documentation):

  1. Load CfRadial polar volume                     (io_radar.read)
  2. Clutter correction: RHOHV gate + spike filter  (clutter_correct.correct)
  3. Velocity dealiasing: region-based              (dealias_velocity.dealias)
  4. Polar -> Cartesian gridding to 81 x 481 x 481  (grid_to_cartesian.grid)
  5. 3D volume rendering with dBZ transfer function (render_dwr.render)

Each intermediate is saved so you can inspect or re-run later steps without
redoing the heavy ones.

Usage:
    python src/pipeline.py data/RCTLS_09JUN2026_050131_L2B_STD.nc
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import io_radar
import clutter_correct
import dealias_velocity
import grid_to_cartesian
import render_dwr


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"


def run(polar_nc: Path) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    stem = polar_nc.stem
    gridded_nc = OUTPUT_DIR / f"{stem}_gridded.nc"
    final_png = OUTPUT_DIR / f"{stem}_3d.png"

    t0 = time.time()

    print(f"\n[1/5] Reading polar volume from {polar_nc.name}...")
    radar = io_radar.read(polar_nc)
    print(f"      {radar.nsweeps} sweeps, {radar.nrays} rays, {radar.ngates} gates")

    print(f"\n[2/5] Clutter correction...")
    radar = clutter_correct.correct(radar)

    print(f"\n[3/5] Velocity dealiasing...")
    radar = dealias_velocity.dealias(radar)

    print(f"\n[4/5] Gridding polar -> Cartesian (this is the slow step)...")
    grid_to_cartesian.grid(radar, gridded_nc)

    print(f"\n[5/5] 3D volume rendering...")
    render_dwr.render(gridded_nc, final_png)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s.")
    print(f"Gridded cube : {gridded_nc}")
    print(f"3D image     : {final_png}")
    return final_png


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python src/pipeline.py <path-to-polar-nc>")
        sys.exit(1)
    run(Path(sys.argv[1]))
