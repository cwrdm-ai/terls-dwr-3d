"""Stacked-CAPPI 3D render — the dark IMD/MOSDAC presentation style.

Reproduces the reference figure: discrete constant-altitude slices (CAPPIs)
of the reflectivity cube drawn as smooth filled-contour sheets stacked in 3D,
on a black background, jet colormap, vertical dBZ colorbar on the right,
lat/lon on the horizontal axes and altitude (km) on the vertical axis.

Rendering style: each sheet is a matplotlib filled contour (contourf) drawn
at its altitude offset — continuous colored regions like the pyiwr / IMD
figures, not per-cell scatter dots. A NaN-aware Gaussian pre-smoothing pass
(SMOOTH_SIGMA) removes single-cell speckle so contours stay clean.

Works with both grid flavors:
  - our pyart cubes  (coords z/y/x in meters from the radar)
  - MOSDAC L2C files (height in m + latitude/longitude in degrees;
    raw data layout (height, lat, lon) — their lon/lat dim names are swapped)

Usage:
    python src/render_stacked_cappi.py <gridded.nc> [layer_step] [dbz_floor]
      layer_step : plot every Nth 250 m level (default 8 = one sheet per 2 km)
      dbz_floor  : hide cells below this dBZ (default 0)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter

# TERLS radar site
TERLS_LAT = 8.5374
TERLS_LON = 76.8657

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

# Color range chosen so the weak-echo bulk (10-25 dBZ) lands in blue/cyan and
# only convective cores reach yellow/red — matching the reference figure.
DBZ_VMIN = 0.0
DBZ_VMAX = 60.0
CONTOUR_LEVELS = np.arange(DBZ_VMIN, DBZ_VMAX + 2.5, 2.5)

# Gaussian smoothing applied per sheet before contouring (grid cells).
# 0 = off. ~1.2 removes single-cell speckle without fattening storm cores.
SMOOTH_SIGMA = 1.2

VIEW_ELEV = 24
VIEW_AZIM = -62


def load_any(nc_path: Path):
    """Return (dbz cube (z, lat, lon), altitudes km, lat axis deg, lon axis deg)."""
    ds = xr.open_dataset(nc_path, decode_times=False)
    dbz = np.asarray(ds["DBZ"].values, dtype=np.float32)
    if dbz.ndim == 4:
        dbz = dbz[0]

    if "z" in ds.variables:            # our pyart flavor — meters from radar
        z_km = np.asarray(ds["z"].values, dtype=np.float32) / 1000.0
        y_km = np.asarray(ds["y"].values, dtype=np.float32) / 1000.0
        x_km = np.asarray(ds["x"].values, dtype=np.float32) / 1000.0
        lat_axis = TERLS_LAT + y_km / 111.0
        lon_axis = TERLS_LON + x_km / (111.0 * np.cos(np.radians(TERLS_LAT)))
    else:                               # MOSDAC L2C flavor — degrees + height
        z_km = np.asarray(ds["height"].values, dtype=np.float32) / 1000.0
        lat_axis = np.asarray(ds["latitude"].values, dtype=np.float32)
        lon_axis = np.asarray(ds["longitude"].values, dtype=np.float32)
    ds.close()
    return dbz, z_km, lat_axis, lon_axis


def smooth_sheet(sheet: np.ndarray, sigma: float) -> np.ndarray:
    """NaN-aware Gaussian smoothing: smooths values without bleeding NaN in,
    and only extends slightly past the original echo edges."""
    valid = np.isfinite(sheet)
    if sigma <= 0 or not valid.any():
        return sheet
    filled = np.where(valid, sheet, 0.0)
    num = gaussian_filter(filled, sigma)
    den = gaussian_filter(valid.astype(np.float32), sigma)
    out = np.full_like(sheet, np.nan)
    ok = den > 0.3        # keep cells with enough real-data support nearby
    out[ok] = num[ok] / den[ok]
    return out


def render(nc_path: Path, layer_step: int, dbz_floor: float, out_png: Path) -> Path:
    print(f"Loading {nc_path.name}...")
    dbz, z_km, lat_axis, lon_axis = load_any(nc_path)
    LON2, LAT2 = np.meshgrid(lon_axis, lat_axis)   # (lat, lon) grids

    layer_idx = list(range(1, dbz.shape[0], layer_step))
    print(f"  contouring {len(layer_idx)} CAPPI sheets "
          f"({z_km[layer_idx[0]]:.2f} -> {z_km[layer_idx[-1]]:.2f} km, "
          f"every {layer_step * 0.25:.1f} km), floor {dbz_floor:.0f} dBZ, "
          f"smooth sigma={SMOOTH_SIGMA}")

    # ---- dark style ----
    fig = plt.figure(figsize=(11, 8.5), facecolor="black")
    ax = fig.add_subplot(111, projection="3d", facecolor="black")
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.set_pane_color((0, 0, 0, 1))
        pane._axinfo["grid"]["color"] = (0.35, 0.35, 0.35, 0.35)

    norm = Normalize(vmin=DBZ_VMIN, vmax=DBZ_VMAX)
    n_drawn = 0
    lat_min = lon_min = np.inf
    lat_max = lon_max = -np.inf
    for k in layer_idx:
        sheet = smooth_sheet(dbz[k], SMOOTH_SIGMA)
        sheet = np.where(sheet >= dbz_floor, sheet, np.nan)
        masked = np.ma.masked_invalid(sheet)
        if masked.count() == 0:
            continue
        valid = np.isfinite(sheet)
        lat_min = min(lat_min, LAT2[valid].min()); lat_max = max(lat_max, LAT2[valid].max())
        lon_min = min(lon_min, LON2[valid].min()); lon_max = max(lon_max, LON2[valid].max())
        ax.contourf(
            LON2, LAT2, masked,
            levels=CONTOUR_LEVELS, cmap="jet", norm=norm,
            offset=float(z_km[k]), zdir="z",
            alpha=0.9, antialiased=True,
        )
        n_drawn += 1
    print(f"  {n_drawn} sheets drawn")

    # Frame the data, not the full 480 km grid — fills the figure like the
    # reference instead of leaving dead space around small storms.
    if np.isfinite(lat_min):
        pad = 0.15
        ax.set_xlim(lon_min - pad, lon_max + pad)
        ax.set_ylim(lat_min - pad, lat_max + pad)

    ax.set_xlabel("Longitude (°E)", color="white", labelpad=8)
    ax.set_ylabel("Latitude (°N)", color="white", labelpad=8)
    ax.set_zlabel("Altitude (km)", color="white", labelpad=6)
    ax.set_zlim(0, 21)
    ax.set_zticks([0, 3.5, 7, 10.5, 14, 17.5, 21])
    ax.tick_params(colors="white", labelsize=8)
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
    # Stretch the vertical screen axis so inter-sheet gaps read clearly.
    ax.set_box_aspect((1.0, 1.0, 1.25))

    sm = cm.ScalarMappable(norm=norm, cmap="jet")
    cb = fig.colorbar(sm, ax=ax, shrink=0.65, pad=0.08)
    cb.set_label("Reflectivity (dBZ)", color="white")
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.get_yticklabels(), color="white")

    ax.set_title(
        f"Stacked CAPPI — TERLS DWR\n{nc_path.stem.replace('_gridded', '')}",
        color="white", fontsize=10,
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight",
                facecolor="black", edgecolor="none")
    plt.close(fig)
    print(f"Saved: {out_png}")
    return out_png


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python src/render_stacked_cappi.py <gridded.nc> [layer_step] [dbz_floor]")
        sys.exit(1)
    nc = Path(sys.argv[1])
    step = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    floor = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    out = OUTPUT_DIR / (nc.stem.replace("_gridded", "") + "_stacked_cappi.png")
    render(nc, step, floor, out)
