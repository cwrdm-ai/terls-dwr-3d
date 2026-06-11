"""Smooth stacked-CAPPI 3D render — filled contours per altitude layer.

The scatter-based version (render_stacked_cappi.py) draws each data cell as a
square marker: honest but pixelated. This version draws each CAPPI sheet as a
FILLED CONTOUR (matplotlib's contourf projected onto its altitude plane), the
same technique behind the smooth look of IMD/pyiwr figures:

  1. NaN-aware Gaussian smoothing of each sheet (weight-normalized so echo
     edges don't bleed into empty sky)
  2. contourf(..., zdir='z', offset=altitude) — a smooth filled polygon layer
  3. auto-zoom onto the storm bounding box so a compact cell fills the frame

Works with both grid flavors (our pyart cubes and MOSDAC L2C files).

Usage:
    python src/render_stacked_smooth.py <gridded.nc> [layer_step] [dbz_floor]
      layer_step : plot every Nth 250 m level (default 8 = one sheet per 2 km)
      dbz_floor  : hide echo below this dBZ (default 10 — cuts the weak haze)
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

# Color range tuned so weak echo lands blue/cyan and cores go yellow/red,
# matching the reference figure's mood.
DBZ_VMIN = 0.0
DBZ_VMAX = 60.0
CONTOUR_LEVELS = np.linspace(DBZ_VMIN, DBZ_VMAX, 25)

SMOOTH_SIGMA = 1.2        # grid cells (~1.2 km): smooth edges but keep core intensity
COVERAGE_CUTOFF = 0.35    # don't extrapolate contours far past real echo
ZOOM_DBZ = 25.0           # auto-zoom to the bbox of echo above this
ZOOM_PAD_CELLS = 35       # padding (km) around that bbox

VIEW_ELEV = 17
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
    else:                               # MOSDAC L2C — degrees + height (raw
        z_km = np.asarray(ds["height"].values, dtype=np.float32) / 1000.0  # layout
        lat_axis = np.asarray(ds["latitude"].values, dtype=np.float32)    # is
        lon_axis = np.asarray(ds["longitude"].values, dtype=np.float32)   # (h,lat,lon))
    ds.close()
    return dbz, z_km, lat_axis, lon_axis


def nan_smooth(sheet: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing that ignores NaN and won't smear echo into the void."""
    finite = np.isfinite(sheet)
    if not finite.any():
        return np.full_like(sheet, np.nan)
    values = gaussian_filter(np.where(finite, sheet, 0.0), sigma)
    weight = gaussian_filter(finite.astype(np.float32), sigma)
    out = values / np.maximum(weight, 1e-6)
    out[weight < COVERAGE_CUTOFF] = np.nan
    return out


def storm_bbox(dbz: np.ndarray, dbz_zoom: float, pad: int):
    """Index bounds (lat0, lat1, lon0, lon1) of the LARGEST connected storm.

    Taking the bbox of *all* strong echo fails when stray strong cells are
    scattered across the domain (the frame stays at full 480 km and the storm
    looks tiny). Instead: column-max the cube, threshold, label the connected
    regions, and zoom onto the one with the greatest integrated intensity.
    """
    from scipy import ndimage

    colmax = np.nanmax(np.where(np.isfinite(dbz), dbz, -99.0), axis=0)
    mask2d = colmax >= dbz_zoom
    if not mask2d.any():
        return 0, dbz.shape[1], 0, dbz.shape[2]

    labels, n = ndimage.label(mask2d, structure=np.ones((3, 3), dtype=bool))
    # Rank clusters by integrated intensity (sum of column-max dBZ above the
    # threshold), not raw area — favors the real storm complex over broad
    # marginal echo.
    scores = ndimage.sum(np.where(mask2d, colmax - dbz_zoom, 0.0),
                         labels, index=range(1, n + 1))
    best = int(np.argmax(scores)) + 1
    lat_idx, lon_idx = np.where(labels == best)

    lat0 = max(int(lat_idx.min()) - pad, 0)
    lat1 = min(int(lat_idx.max()) + pad, dbz.shape[1])
    lon0 = max(int(lon_idx.min()) - pad, 0)
    lon1 = min(int(lon_idx.max()) + pad, dbz.shape[2])
    return lat0, lat1, lon0, lon1


def render(nc_path: Path, layer_step: int, dbz_floor: float, out_png: Path) -> Path:
    print(f"Loading {nc_path.name}...")
    dbz, z_km, lat_axis, lon_axis = load_any(nc_path)

    lat0, lat1, lon0, lon1 = storm_bbox(dbz, ZOOM_DBZ, ZOOM_PAD_CELLS)
    dbz = dbz[:, lat0:lat1, lon0:lon1]
    lat_axis, lon_axis = lat_axis[lat0:lat1], lon_axis[lon0:lon1]
    print(f"  zoomed to storm: lon [{lon_axis.min():.2f}, {lon_axis.max():.2f}], "
          f"lat [{lat_axis.min():.2f}, {lat_axis.max():.2f}]")

    LON2, LAT2 = np.meshgrid(lon_axis, lat_axis)
    layer_idx = [k for k in range(1, dbz.shape[0], layer_step)]

    # ---- dark style ----
    fig = plt.figure(figsize=(11, 8.5), facecolor="black")
    ax = fig.add_subplot(111, projection="3d", facecolor="black")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0, 0, 0, 1))
        axis._axinfo["grid"]["color"] = (0.35, 0.35, 0.35, 0.35)

    norm = Normalize(vmin=DBZ_VMIN, vmax=DBZ_VMAX)
    n_drawn = 0
    for k in layer_idx:
        smooth = nan_smooth(dbz[k], SMOOTH_SIGMA)
        smooth = np.where(smooth >= dbz_floor, smooth, np.nan)
        if not np.isfinite(smooth).any():
            continue
        ax.contourf(
            LON2, LAT2, np.ma.masked_invalid(smooth),
            levels=CONTOUR_LEVELS, cmap="jet", norm=norm,
            zdir="z", offset=float(z_km[k]), alpha=0.9, antialiased=True,
        )
        n_drawn += 1
    print(f"  {n_drawn} smooth CAPPI sheets drawn "
          f"(every {layer_step * 0.25:.1f} km, floor {dbz_floor:.0f} dBZ)")

    ax.set_xlabel("Longitude (°E)", color="white", labelpad=10)
    ax.set_ylabel("Latitude (°N)", color="white", labelpad=10)
    ax.set_zlabel("Altitude (km)", color="white", labelpad=6)
    ax.set_zlim(0, 21)
    ax.set_zticks([0, 3.5, 7, 10.5, 14, 17.5, 21])
    ax.tick_params(colors="white", labelsize=8)
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)

    mappable = cm.ScalarMappable(norm=norm, cmap="jet")
    cb = fig.colorbar(mappable, ax=ax, shrink=0.62, pad=0.08)
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
        print("usage: python src/render_stacked_smooth.py <gridded.nc> [layer_step] [dbz_floor]")
        sys.exit(1)
    nc = Path(sys.argv[1])
    step = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    floor = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
    out = OUTPUT_DIR / (nc.stem.replace("_gridded", "") + "_stacked_smooth.png")
    render(nc, step, floor, out)
