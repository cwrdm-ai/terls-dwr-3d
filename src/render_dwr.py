"""Render a gridded 3D DWR reflectivity cube in MOSDAC-style.

Inputs the NetCDF written by grid_to_cartesian (a pyart.Grid file). That file's
DBZ variable has shape (z, y, x) where:
    z is altitude in meters (0 to 20,000 m, 81 levels)
    y is north-south distance from the radar in meters
    x is east-west distance from the radar in meters

Pipeline here:
  1. Load gridded NetCDF
  2. Wrap as pyvista.ImageData with physical spacing
  3. Apply dBZ -> RGBA transfer function (see transfer_function.py)
  4. Volume-render and screenshot to output/
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyvista as pv
import xarray as xr

from transfer_function import build_dbz_transfer_function

# TERLS radar site, Thumba, Kerala
TERLS_LAT = 8.5374
TERLS_LON = 76.8657

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

# Vertical exaggeration. Real atmosphere is wide and thin: a 100 km storm is
# only 15 km tall. With 1:1 aspect ratio storms look like pancakes. MOSDAC's
# hero shots typically stretch vertical 3x-5x.
VERTICAL_EXAGGERATION = 4.0


def load_gridded(nc_path: Path) -> tuple[np.ndarray, dict]:
    """Read the pyart-written gridded NetCDF and return (DBZ cube, metadata)."""
    ds = xr.open_dataset(nc_path, decode_times=False)

    # pyart.io.write_grid uses standard CF names: z, y, x (in meters).
    dbz = np.asarray(ds["DBZ"].values, dtype=np.float32)

    # pyart writes the field with a time dimension of size 1 — squeeze it.
    if dbz.ndim == 4 and dbz.shape[0] == 1:
        dbz = dbz[0]

    # Replace fill values with NaN.
    fill = ds["DBZ"].attrs.get("_FillValue")
    if fill is not None:
        dbz[dbz == fill] = np.nan

    meta = {
        "z_m": np.asarray(ds["z"].values, dtype=np.float32),
        "y_m": np.asarray(ds["y"].values, dtype=np.float32),
        "x_m": np.asarray(ds["x"].values, dtype=np.float32),
        "shape": dbz.shape,
    }
    ds.close()
    return dbz, meta


def to_image_data(dbz_zyx: np.ndarray, meta: dict) -> pv.ImageData:
    """Wrap the (Z, Y, X) cube as a uniformly-spaced PyVista grid.

    Spacing reflects the physical grid: 1 km horizontal, 0.25 km vertical
    (post vertical-exaggeration: VERTICAL_EXAGGERATION * 0.25 km).
    PyVista expects Fortran-order (X, Y, Z) ravel.
    """
    nz, ny, nx = dbz_zyx.shape
    dx_km = (meta["x_m"][1] - meta["x_m"][0]) / 1000.0
    dy_km = (meta["y_m"][1] - meta["y_m"][0]) / 1000.0
    dz_km = (meta["z_m"][1] - meta["z_m"][0]) / 1000.0

    data_xyz = np.transpose(dbz_zyx, (2, 1, 0))  # (Z,Y,X) -> (X,Y,Z)

    grid = pv.ImageData()
    grid.dimensions = (nx, ny, nz)
    grid.spacing = (dx_km, dy_km, dz_km * VERTICAL_EXAGGERATION)
    # Place origin at the radar (the cube is centered on radar in x/y).
    grid.origin = (
        float(meta["x_m"][0]) / 1000.0,
        float(meta["y_m"][0]) / 1000.0,
        0.0,
    )
    grid.point_data["dBZ"] = data_xyz.ravel(order="F").astype(np.float32)
    return grid


def render(gridded_nc: Path, screenshot: Path) -> Path:
    print(f"Loading gridded cube: {gridded_nc}")
    dbz, meta = load_gridded(gridded_nc)
    finite = dbz[np.isfinite(dbz)]
    if finite.size:
        print(f"  shape={dbz.shape}  dBZ range=[{finite.min():.1f}, {finite.max():.1f}]")
    else:
        print("  WARNING: gridded cube has no finite DBZ values!")

    pv_grid = to_image_data(dbz, meta)
    color_tf, opacity_tf = build_dbz_transfer_function()

    plotter = pv.Plotter(off_screen=True, window_size=(1600, 1200))
    plotter.set_background("black")
    plotter.add_volume(
        pv_grid,
        scalars="dBZ",
        cmap=color_tf,
        opacity=opacity_tf,
        clim=(0, 70),
        shade=True,
        ambient=0.3,
        diffuse=0.7,
        specular=0.2,
        scalar_bar_args={"title": "Reflectivity (dBZ)", "color": "white"},
    )

    plotter.add_bounding_box(color="gray", line_width=1, opacity=0.4)
    plotter.show_axes()
    plotter.add_text(
        f"TERLS C-Band DWR\nVertical exag x{VERTICAL_EXAGGERATION:.0f}",
        position="upper_left",
        color="white",
        font_size=10,
    )
    plotter.camera_position = "iso"
    plotter.camera.zoom(1.2)

    OUTPUT_DIR.mkdir(exist_ok=True)
    plotter.screenshot(str(screenshot))
    print(f"Saved: {screenshot}")
    return screenshot


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python src/render_dwr.py <path-to-gridded-nc>")
        sys.exit(1)
    nc = Path(sys.argv[1])
    out = OUTPUT_DIR / (nc.stem + "_3d.png")
    render(nc, out)
