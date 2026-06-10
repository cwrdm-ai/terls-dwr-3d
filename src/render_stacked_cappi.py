"""Stacked-CAPPI 3D render — the dark IMD/MOSDAC presentation style.

Reproduces the reference figure: discrete constant-altitude slices (CAPPIs)
of the reflectivity cube drawn as flat speckled sheets stacked in 3D, on a
black background, jet colormap, vertical dBZ colorbar on the right, lat/lon
on the horizontal axes and altitude (km) on the vertical axis.

Why stacked layers instead of a continuous cloud: each sheet reads as a map
("what does the rain field look like at 4 km?") while the stack shows the
vertical structure — where echo tops sit, how cores tilt with height. It is
the operational radar-analyst's view of the same 81 x 481 x 481 cube.

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
from matplotlib.colors import Normalize

# TERLS radar site
TERLS_LAT = 8.5374
TERLS_LON = 76.8657

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

# Color range chosen so the weak-echo bulk (10-25 dBZ) lands in blue/cyan and
# only convective cores reach yellow/red — matching the reference figure.
DBZ_VMIN = 0.0
DBZ_VMAX = 60.0
MAX_POINTS_PER_LAYER = 25_000
VIEW_ELEV = 18
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


def render(nc_path: Path, layer_step: int, dbz_floor: float, out_png: Path) -> Path:
    print(f"Loading {nc_path.name}...")
    dbz, z_km, lat_axis, lon_axis = load_any(nc_path)
    LON2, LAT2 = np.meshgrid(lon_axis, lat_axis)   # (lat, lon) grids

    layer_idx = list(range(1, dbz.shape[0], layer_step))
    print(f"  plotting {len(layer_idx)} CAPPI sheets "
          f"({z_km[layer_idx[0]]:.2f} -> {z_km[layer_idx[-1]]:.2f} km, "
          f"every {layer_step * 0.25:.1f} km), floor {dbz_floor:.0f} dBZ")

    # ---- dark style ----
    fig = plt.figure(figsize=(10, 8), facecolor="black")
    ax = fig.add_subplot(111, projection="3d", facecolor="black")
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.set_pane_color((0, 0, 0, 1))
        pane._axinfo["grid"]["color"] = (0.35, 0.35, 0.35, 0.35)

    norm = Normalize(vmin=DBZ_VMIN, vmax=DBZ_VMAX)
    rng = np.random.default_rng(11)
    sc = None
    total = 0
    for k in layer_idx:
        sheet = dbz[k]
        keep = np.isfinite(sheet) & (sheet >= dbz_floor)
        n = int(keep.sum())
        if n == 0:
            continue
        lons, lats, vals = LON2[keep], LAT2[keep], sheet[keep]
        if n > MAX_POINTS_PER_LAYER:
            sel = rng.choice(n, MAX_POINTS_PER_LAYER, replace=False)
            lons, lats, vals = lons[sel], lats[sel], vals[sel]
        total += lons.size
        sc = ax.scatter(
            lons, lats, np.full(lons.size, z_km[k]),
            c=vals, cmap="jet", norm=norm,
            s=2.5, marker="s", alpha=0.8, linewidths=0, depthshade=False,
        )
    print(f"  {total:,} points drawn")

    ax.set_xlabel("Longitude (°E)", color="white", labelpad=8)
    ax.set_ylabel("Latitude (°N)", color="white", labelpad=8)
    ax.set_zlabel("Altitude (km)", color="white", labelpad=6)
    ax.set_zlim(0, 21)
    ax.set_zticks([0, 3.5, 7, 10.5, 14, 17.5, 21])
    ax.tick_params(colors="white", labelsize=8)
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)

    if sc is not None:
        cb = fig.colorbar(sc, ax=ax, shrink=0.65, pad=0.08)
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
