"""Diagnostic: plot the RAW polar sweeps in 3D to reveal elevation structure.

Purpose — answer the question "is the IMD figure showing different elevation
angles?". A Doppler radar volume scan is built from N discrete elevation
sweeps (TERLS uses 11: 0.5, 1, 2, 3, 4, 7, 9, 12, 15, 18, 21 degrees). Each
sweep is a 360-degree azimuthal rotation at a FIXED tilt, so in 3D space each
sweep traces a CONE: the beam rises with range. Stack all 11 and you get
nested cones — the "umbrella ribs" signature.

If a 3D radar figure shows that conical / arced layering with gaps between
shells, it is plotting POLAR sweep data (you can literally see the elevations).
If it shows a smooth filled volume with no conical seams, it is the GRIDDED
Cartesian product (elevations interpolated away).

We render two versions of the SAME polar data:
  A) colored by SWEEP INDEX  -> makes the 11 cones obvious
  B) colored by dBZ          -> to compare directly with the IMD reference

pyart gives us gate_x / gate_y / gate_z (true beam geometry incl. 4/3-earth
refraction), so the cone heights are physically correct.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import io_radar


OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

DBZ_THRESHOLD = 5.0          # only plot gates above this (cut clear-air noise)
MAX_POINTS = 70_000          # matplotlib 3D scatter budget
ALT_MAX_KM = 21.0


def _sweep_index_per_ray(radar) -> np.ndarray:
    idx = np.zeros(radar.nrays, dtype=int)
    for i, sl in enumerate(radar.iter_slice()):
        idx[sl] = i
    return idx


def render(polar_nc: Path) -> None:
    print(f"Loading polar volume {polar_nc.name}...")
    radar = io_radar.read(polar_nc)

    # Physical Cartesian position of every gate (meters), incl. beam curvature.
    gx = radar.gate_x["data"] / 1000.0   # (nrays, ngates) km, E-W
    gy = radar.gate_y["data"] / 1000.0   # km, N-S
    gz = radar.gate_z["data"] / 1000.0   # km, altitude above radar

    dbz = np.ma.getdata(radar.fields["DBZ"]["data"]).astype(np.float32)
    dbz_mask = np.ma.getmaskarray(radar.fields["DBZ"]["data"])

    sweep_idx_ray = _sweep_index_per_ray(radar)
    sweep_idx = np.repeat(sweep_idx_ray[:, None], radar.ngates, axis=1)
    elev_per_ray = radar.elevation["data"].astype(np.float32)
    elev = np.repeat(elev_per_ray[:, None], radar.ngates, axis=1)

    # Flatten and select.
    keep = (~dbz_mask) & np.isfinite(dbz) & (dbz >= DBZ_THRESHOLD) & (gz <= ALT_MAX_KM)
    fx, fy, fz = gx[keep], gy[keep], gz[keep]
    fdbz = dbz[keep]
    fsweep = sweep_idx[keep]
    felev = elev[keep]
    print(f"  gates above {DBZ_THRESHOLD} dBZ: {fdbz.size:,}")

    if fdbz.size > MAX_POINTS:
        rng = np.random.default_rng(0)
        sel = rng.choice(fdbz.size, MAX_POINTS, replace=False)
        fx, fy, fz, fdbz, fsweep, felev = (
            fx[sel], fy[sel], fz[sel], fdbz[sel], fsweep[sel], felev[sel]
        )
        print(f"  downsampled to {fdbz.size:,}")

    elev_angles = [float(a) for a in radar.fixed_angle["data"]]
    print(f"  elevation angles: {elev_angles}")

    # ---- Figure A: colored by elevation sweep (reveals the cones) ----
    figA = plt.figure(figsize=(9, 7), facecolor="white")
    axA = figA.add_subplot(111, projection="3d")
    scA = axA.scatter(fx, fy, fz, c=felev, cmap="turbo", s=3, alpha=0.5, linewidths=0)
    axA.set_xlabel("E–W distance (km)")
    axA.set_ylabel("N–S distance (km)")
    axA.set_zlabel("Altitude (km)")
    axA.set_zlim(0, ALT_MAX_KM)
    axA.set_title("Polar sweeps colored by ELEVATION angle\n(each color = one of 11 sweep cones)")
    cbA = figA.colorbar(scA, ax=axA, shrink=0.6, pad=0.1)
    cbA.set_label("Elevation angle (°)")
    axA.view_init(elev=12, azim=-60)
    outA = OUTPUT_DIR / (polar_nc.stem + "_polar_by_elevation.png")
    figA.savefig(outA, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(figA)
    print(f"Saved: {outA}")

    # ---- Figure B: same data colored by dBZ (compare with IMD reference) ----
    figB = plt.figure(figsize=(9, 7), facecolor="white")
    axB = figB.add_subplot(111, projection="3d")
    scB = axB.scatter(fx, fy, fz, c=fdbz, cmap="jet", vmin=-25, vmax=60,
                      s=3, alpha=0.55, linewidths=0)
    axB.set_xlabel("E–W distance (km)")
    axB.set_ylabel("N–S distance (km)")
    axB.set_zlabel("Altitude (km)")
    axB.set_zlim(0, ALT_MAX_KM)
    axB.set_title("Same polar data colored by dBZ\n(compare structure with IMD reference)")
    cbB = figB.colorbar(scB, ax=axB, shrink=0.6, pad=0.1)
    cbB.set_label("Reflectivity (dBZ)")
    axB.view_init(elev=12, azim=-60)
    outB = OUTPUT_DIR / (polar_nc.stem + "_polar_by_dbz.png")
    figB.savefig(outB, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(figB)
    print(f"Saved: {outB}")


if __name__ == "__main__":
    nc = sys.argv[1] if len(sys.argv) > 1 else "data/RCTLS_09JUN2026_050131_L2B_STD.nc"
    render(Path(nc))
