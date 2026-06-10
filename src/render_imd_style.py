"""IMD/MOSDAC-style 3D DWR render — matches the operational visual format.

Visual choices that distinguish this from the cinematic render:

  - **White background** + **jet colormap** (NWS/IMD operational convention).
  - **Lat/lon axes in degrees**, not km-from-radar. The viewer can read
    "this cell is at 76.85°E, 8.55°N, 4 km altitude" right off the plot.
  - **Discrete voxel display** instead of volumetric ray-march — preserves
    each measurement cell as a visible cube. This is the matplotlib
    Axes3D / Mayavi look that IMD publishes.
  - **Vertical colorbar on the right** with dBZ scale.
  - **Altitude labeled 0-21 km** vertically.

We use matplotlib's 3D scatter (one dot per cell) which is the cheapest way
to reproduce the discrete-cell look. For ~100k+ cells, scatter is fast and
gives the chunky voxel appearance the IMD reference shows.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import Normalize


# TERLS radar site
TERLS_LAT = 8.5374
TERLS_LON = 76.8657

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

# dBZ display range — IMD typically shows -25 to +50 (or similar).
# We use a slightly wider range to catch any heavy cores.
DBZ_DISPLAY_MIN = -25.0
DBZ_DISPLAY_MAX = 60.0

# Only show cells with dBZ above this — below this is noise / very light precip
# that would just clutter the plot. Raise to make plot sparser.
DBZ_PLOT_THRESHOLD = 0.0

# At ~1.1M valid cells the plot would be unreadable. Downsample.
MAX_POINTS_TO_PLOT = 80_000


def km_to_deg_lat(km: float) -> float:
    """Convert N-S km to degrees of latitude. 1 deg lat ≈ 111 km."""
    return km / 111.0


def km_to_deg_lon(km: float, lat_deg: float) -> float:
    """Convert E-W km to degrees of longitude at a given latitude."""
    return km / (111.0 * np.cos(np.radians(lat_deg)))


def render_imd(gridded_nc: Path, out_png: Path) -> Path:
    print(f"Loading {gridded_nc.name}...")
    ds = xr.open_dataset(gridded_nc, decode_times=False)
    dbz = np.asarray(ds["DBZ"].values, dtype=np.float32)
    if dbz.ndim == 4:
        dbz = dbz[0]

    z_m = np.asarray(ds["z"].values, dtype=np.float32)      # altitude m
    y_m = np.asarray(ds["y"].values, dtype=np.float32)      # N-S m from radar
    x_m = np.asarray(ds["x"].values, dtype=np.float32)      # E-W m from radar
    ds.close()

    # Build 3D coordinate arrays then flatten.
    Z, Y, X = np.meshgrid(z_m, y_m, x_m, indexing="ij")
    flat_dbz = dbz.ravel()
    flat_z_km = Z.ravel() / 1000.0
    flat_y_km = Y.ravel() / 1000.0
    flat_x_km = X.ravel() / 1000.0

    # Keep only cells we want to display.
    keep = np.isfinite(flat_dbz) & (flat_dbz >= DBZ_PLOT_THRESHOLD)
    flat_dbz = flat_dbz[keep]
    flat_z_km = flat_z_km[keep]
    flat_y_km = flat_y_km[keep]
    flat_x_km = flat_x_km[keep]

    print(f"  cells to plot before downsample: {flat_dbz.size:,}")

    # Downsample weighted toward higher dBZ so we keep the storm cores.
    if flat_dbz.size > MAX_POINTS_TO_PLOT:
        # Weight by max(0, dBZ - 5) so noise gates have low chance, hail cores
        # have high chance.
        w = np.clip(flat_dbz - 5.0, 0.5, None)
        w = w / w.sum()
        rng = np.random.default_rng(42)  # deterministic
        idx = rng.choice(flat_dbz.size, size=MAX_POINTS_TO_PLOT, replace=False, p=w)
        flat_dbz = flat_dbz[idx]
        flat_z_km = flat_z_km[idx]
        flat_y_km = flat_y_km[idx]
        flat_x_km = flat_x_km[idx]
        print(f"  downsampled to {flat_dbz.size:,} (weighted by intensity)")

    # Convert km offsets to absolute lat/lon degrees.
    lat_deg = TERLS_LAT + km_to_deg_lat(flat_y_km)
    lon_deg = TERLS_LON + km_to_deg_lon(flat_x_km, TERLS_LAT)

    # Plot
    fig = plt.figure(figsize=(10, 8), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")

    norm = Normalize(vmin=DBZ_DISPLAY_MIN, vmax=DBZ_DISPLAY_MAX)
    scatter = ax.scatter(
        lon_deg, lat_deg, flat_z_km,
        c=flat_dbz, cmap="jet", norm=norm,
        s=4, marker="s", alpha=0.55, linewidths=0,
    )

    # Axes labels and limits
    ax.set_xlabel("Longitude (°E)", labelpad=10)
    ax.set_ylabel("Latitude (°N)", labelpad=10)
    ax.set_zlabel("Altitude (km)", labelpad=10)
    ax.set_zlim(0, 21)
    # Trim the lat/lon view to the actual data extent (slightly padded).
    pad = 0.05
    ax.set_xlim(np.percentile(lon_deg, 1) - pad, np.percentile(lon_deg, 99) + pad)
    ax.set_ylim(np.percentile(lat_deg, 1) - pad, np.percentile(lat_deg, 99) + pad)

    # Mark the radar site with a small black square.
    ax.scatter([TERLS_LON], [TERLS_LAT], [0.0],
               c="black", marker="s", s=60, label="TERLS")

    # Colorbar
    cb = fig.colorbar(scatter, ax=ax, pad=0.10, shrink=0.7)
    cb.set_label("Reflectivity (dBZ)")

    # Title / metadata
    scan_id = gridded_nc.stem.replace("_gridded", "")
    ax.set_title(f"3D Volumetric DWR — TERLS  ({scan_id})", pad=15)

    # Match a typical IMD viewing angle.
    ax.view_init(elev=20, azim=-65)

    OUTPUT_DIR.mkdir(exist_ok=True)
    fig.savefig(out_png, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_png}")
    return out_png


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python src/render_imd_style.py <gridded-nc-path>")
        sys.exit(1)
    nc = Path(sys.argv[1])
    out = OUTPUT_DIR / (nc.stem + "_imd.png")
    render_imd(nc, out)
