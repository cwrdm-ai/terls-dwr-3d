"""Reproduce the IMD/MOSDAC 3D DWR presentation as closely as possible.

Matches the reference figure's styling:
  - matplotlib mplot3d box with the default gridded panes (light grey grid)
  - jet colormap
  - lat/lon degree axes (X = longitude E, Y = latitude N)
  - altitude 0-21 km on Z with ticks at 0/7/14/21
  - vertical dBZ colorbar on the right
  - small square voxel markers, semi-transparent
  - side-on-ish viewing angle

All the look-knobs are at the top so we can iterate quickly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import Normalize

# ---- TERLS site ----
TERLS_LAT = 8.5374
TERLS_LON = 76.8657

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

# ============================================================
#                    LOOK KNOBS — tune these
# ============================================================
DBZ_VMIN = -25.0          # colorbar bottom
DBZ_VMAX = 50.0           # colorbar top
DBZ_PLOT_THRESHOLD = 0.0  # only plot cells >= this dBZ
ALT_MAX_KM = 21.0

N_SAMPLE = 130_000        # uniform random sample (preserves layering)
MARKER_SIZE = 3
MARKER = "s"              # square -> voxel look
ALPHA = 0.5

VIEW_ELEV = 15            # camera elevation (deg)
VIEW_AZIM = -60           # camera azimuth (deg)

# Optional lat/lon crop window (set to None for full data extent).
LON_WINDOW = None
LAT_WINDOW = None
# ============================================================


def km_to_deg_lat(km):
    return km / 111.0


def km_to_deg_lon(km, lat_deg):
    return km / (111.0 * np.cos(np.radians(lat_deg)))


def render(gridded_nc: Path, out_png: Path) -> Path:
    """Render either grid flavor:
    - our pyart cubes: coords z/y/x in meters from the radar
    - MOSDAC L2C: coords height (m) + latitude/longitude (deg); data layout is
      (height, lat, lon) despite dims being NAMED (height, lon, lat) — the
      labels are swapped in their writer (verified empirically vs our cube).
    """
    print(f"Loading {gridded_nc.name}...")
    ds = xr.open_dataset(gridded_nc, decode_times=False)
    dbz = np.asarray(ds["DBZ"].values, dtype=np.float32)
    if dbz.ndim == 4:
        dbz = dbz[0]

    if "z" in ds.variables:           # our pyart flavor: meters from radar
        z_m = np.asarray(ds["z"].values, dtype=np.float32)
        y_m = np.asarray(ds["y"].values, dtype=np.float32)
        x_m = np.asarray(ds["x"].values, dtype=np.float32)
        ds.close()

        Z, Y, X = np.meshgrid(z_m, y_m, x_m, indexing="ij")
        fd = dbz.ravel()
        fz = Z.ravel() / 1000.0
        fy = Y.ravel() / 1000.0
        fx = X.ravel() / 1000.0

        keep = np.isfinite(fd) & (fd >= DBZ_PLOT_THRESHOLD) & (fz <= ALT_MAX_KM)
        fd, fz, fy, fx = fd[keep], fz[keep], fy[keep], fx[keep]

        lat = TERLS_LAT + km_to_deg_lat(fy)
        lon = TERLS_LON + km_to_deg_lon(fx, TERLS_LAT)
    else:                              # MOSDAC L2C flavor: degrees + height
        h_m = np.asarray(ds["height"].values, dtype=np.float32)
        lat_axis = np.asarray(ds["latitude"].values, dtype=np.float32)
        lon_axis = np.asarray(ds["longitude"].values, dtype=np.float32)
        ds.close()

        # Raw layout is (height, lat, lon) — see docstring.
        H, LAT, LON = np.meshgrid(h_m, lat_axis, lon_axis, indexing="ij")
        fd = dbz.ravel()
        fz = H.ravel() / 1000.0
        lat = LAT.ravel()
        lon = LON.ravel()

        keep = np.isfinite(fd) & (fd >= DBZ_PLOT_THRESHOLD) & (fz <= ALT_MAX_KM)
        fd, fz, lat, lon = fd[keep], fz[keep], lat[keep], lon[keep]

    if LON_WINDOW is not None:
        m = (lon >= LON_WINDOW[0]) & (lon <= LON_WINDOW[1])
        fd, fz, lat, lon = fd[m], fz[m], lat[m], lon[m]
    if LAT_WINDOW is not None:
        m = (lat >= LAT_WINDOW[0]) & (lat <= LAT_WINDOW[1])
        fd, fz, lat, lon = fd[m], fz[m], lat[m], lon[m]

    print(f"  cells to plot: {fd.size:,}")
    # UNIFORM sample (not intensity-weighted) so horizontal layers are preserved.
    if fd.size > N_SAMPLE:
        rng = np.random.default_rng(7)
        sel = rng.choice(fd.size, N_SAMPLE, replace=False)
        fd, fz, lat, lon = fd[sel], fz[sel], lat[sel], lon[sel]
        print(f"  uniform-sampled to {fd.size:,}")

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    norm = Normalize(vmin=DBZ_VMIN, vmax=DBZ_VMAX)
    sc = ax.scatter(lon, lat, fz, c=fd, cmap="jet", norm=norm,
                    s=MARKER_SIZE, marker=MARKER, alpha=ALPHA, linewidths=0,
                    depthshade=False)

    ax.set_xlabel("Longitude (°E)", labelpad=8)
    ax.set_ylabel("Latitude (°N)", labelpad=8)
    ax.set_zlabel("Altitude (km)", labelpad=6)
    ax.set_zlim(0, ALT_MAX_KM)
    ax.set_zticks([0, 7, 14, 21])

    cb = fig.colorbar(sc, ax=ax, shrink=0.65, pad=0.10)
    cb.set_label("Reflectivity (dBZ)")

    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)

    scan_id = gridded_nc.stem.replace("_gridded", "")
    ax.set_title(f"3D Volumetric DWR (dBZ) — TERLS\n{scan_id}", fontsize=10)

    OUTPUT_DIR.mkdir(exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")
    return out_png


if __name__ == "__main__":
    nc = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        OUTPUT_DIR / "RCTLS_09JUN2026_050131_L2B_STD_gridded.nc"
    if len(sys.argv) > 2:   # optional dBZ display floor, e.g. 10
        DBZ_PLOT_THRESHOLD = float(sys.argv[2])
        suffix = f"_imd_exact_min{sys.argv[2]}dBZ.png"
    else:
        suffix = "_imd_exact.png"
    out = OUTPUT_DIR / (nc.stem.replace("_gridded", "") + suffix)
    render(nc, out)
