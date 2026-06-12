"""Exact recreation of MOSDAC's dark stacked-layer 3D DWR figure.

Forensic styling notes from the reference (mosdac.gov.in 3D Volumetric TERLS
DWR product page):
  - speckled cell-level scatter per altitude layer (NOT smoothed contours)
  - pure black background, no visible pane grids, thin white ticks
  - altitude ticks at 3.5 km multiples: 3.5, 7, 10.5, 14, 17.5, 21
  - SEGMENTED jet colorbar on the right (discrete 5 dBZ bands)
  - no title clutter inside the canvas
  - moderate side-on view, storm stack fills the frame
  - weak echo cyan/blue, cores yellow/orange/red

Works with both grid flavors (our pyart cubes / MOSDAC L2C files).

Usage:
    python src/render_mosdac_exact.py <gridded.nc> [layer_step] [dbz_floor]
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import cm
from matplotlib.colors import BoundaryNorm

TERLS_LAT = 8.5374
TERLS_LON = 76.8657

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

# Discrete 5 dBZ bands, 0 -> 65: gives the segmented colorbar of the reference.
BOUNDS = np.arange(0.0, 70.0, 5.0)

MAX_POINTS_PER_LAYER = 30_000
MARKER_SIZE = 2.0
VIEW_ELEV = 18
VIEW_AZIM = -122   # quadrant chosen so matplotlib draws the altitude axis on the LEFT, as in the reference


def load_any(nc_path: Path):
    ds = xr.open_dataset(nc_path, decode_times=False)
    dbz = np.asarray(ds["DBZ"].values, dtype=np.float32)
    if dbz.ndim == 4:
        dbz = dbz[0]
    if "z" in ds.variables:            # our pyart flavor
        z_km = np.asarray(ds["z"].values, dtype=np.float32) / 1000.0
        y_km = np.asarray(ds["y"].values, dtype=np.float32) / 1000.0
        x_km = np.asarray(ds["x"].values, dtype=np.float32) / 1000.0
        lat_axis = TERLS_LAT + y_km / 111.0
        lon_axis = TERLS_LON + x_km / (111.0 * np.cos(np.radians(TERLS_LAT)))
    else:                               # MOSDAC L2C (raw layout (h, lat, lon))
        z_km = np.asarray(ds["height"].values, dtype=np.float32) / 1000.0
        lat_axis = np.asarray(ds["latitude"].values, dtype=np.float32)
        lon_axis = np.asarray(ds["longitude"].values, dtype=np.float32)
    ds.close()
    return dbz, z_km, lat_axis, lon_axis


def storm_bbox(dbz: np.ndarray, dbz_zoom: float = 25.0, pad: int = 30):
    """Frame the largest connected storm complex (by integrated intensity)."""
    from scipy import ndimage

    colmax = np.nanmax(np.where(np.isfinite(dbz), dbz, -99.0), axis=0)
    mask2d = colmax >= dbz_zoom
    if not mask2d.any():
        return 0, dbz.shape[1], 0, dbz.shape[2]
    labels, n = ndimage.label(mask2d, structure=np.ones((3, 3), dtype=bool))
    scores = ndimage.sum(np.where(mask2d, colmax - dbz_zoom, 0.0),
                         labels, index=range(1, n + 1))
    best = int(np.argmax(scores)) + 1
    lat_idx, lon_idx = np.where(labels == best)
    return (max(int(lat_idx.min()) - pad, 0), min(int(lat_idx.max()) + pad, dbz.shape[1]),
            max(int(lon_idx.min()) - pad, 0), min(int(lon_idx.max()) + pad, dbz.shape[2]))


def render(nc_path: Path, layer_step: int, dbz_floor: float, out_png: Path) -> Path:
    print(f"Loading {nc_path.name}...")
    dbz, z_km, lat_axis, lon_axis = load_any(nc_path)

    lat0, lat1, lon0, lon1 = storm_bbox(dbz)
    dbz = dbz[:, lat0:lat1, lon0:lon1]
    lat_axis, lon_axis = lat_axis[lat0:lat1], lon_axis[lon0:lon1]
    print(f"  framed: lon [{lon_axis.min():.2f}, {lon_axis.max():.2f}], "
          f"lat [{lat_axis.min():.2f}, {lat_axis.max():.2f}]")

    LON2, LAT2 = np.meshgrid(lon_axis, lat_axis)
    cmap = plt.get_cmap("jet")
    norm = BoundaryNorm(BOUNDS, cmap.N)
    rng = np.random.default_rng(3)

    fig = plt.figure(figsize=(10, 7.5), facecolor="black")
    ax = fig.add_subplot(111, projection="3d", facecolor="black")
    # Reference shows pure black space — kill the pane fills and grid lines.
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0, 0, 0, 1))
        axis._axinfo["grid"]["color"] = (0, 0, 0, 0)

    total = 0
    for k in range(1, dbz.shape[0], layer_step):
        sheet = dbz[k]
        keep = np.isfinite(sheet) & (sheet >= dbz_floor)
        n = int(keep.sum())
        if n == 0:
            continue
        lons, lats, vals = LON2[keep], LAT2[keep], sheet[keep]
        if n > MAX_POINTS_PER_LAYER:
            sel = rng.choice(n, MAX_POINTS_PER_LAYER, replace=False)
            lons, lats, vals = lons[sel], lats[sel], vals[sel]
        ax.scatter(lons, lats, np.full(lons.size, z_km[k]),
                   c=vals, cmap=cmap, norm=norm,
                   s=MARKER_SIZE, marker="s", alpha=0.95,
                   linewidths=0, depthshade=False)
        total += lons.size
    print(f"  {total:,} cells drawn")

    ax.set_zlim(0, 21)
    ax.set_zticks([3.5, 7, 10.5, 14, 17.5, 21])
    ax.tick_params(colors="white", labelsize=7, pad=-2)
    ax.set_xlabel("Longitude (°E)", color="white", fontsize=8, labelpad=4)
    ax.set_ylabel("Latitude (°N)", color="white", fontsize=8, labelpad=4)
    ax.set_zlabel("Altitude (km)", color="white", fontsize=8, labelpad=2)
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
    # Tighten the camera on the data.
    ax.set_box_aspect((1.15, 1.0, 0.85), zoom=1.12)

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(mappable, ax=ax, shrink=0.78, pad=0.09,
                      boundaries=BOUNDS, ticks=BOUNDS[::2])
    cb.ax.yaxis.set_tick_params(color="white", labelsize=7)
    plt.setp(cb.ax.get_yticklabels(), color="white")
    cb.outline.set_edgecolor("white")
    cb.outline.set_linewidth(0.5)

    OUTPUT_DIR.mkdir(exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight",
                facecolor="black", edgecolor="none")
    plt.close(fig)
    print(f"Saved: {out_png}")
    return out_png


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python src/render_mosdac_exact.py <gridded.nc> [layer_step] [dbz_floor]")
        sys.exit(1)
    nc = Path(sys.argv[1])
    step = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    floor = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
    out = OUTPUT_DIR / (nc.stem.replace("_gridded", "") + "_mosdac_exact.png")
    render(nc, step, floor, out)
