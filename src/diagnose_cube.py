"""Quick diagnostic on the gridded cube — figure out why the render is empty.

We answer three questions:
  1. What's the actual dBZ distribution? (histogram)
  2. Where in the cube is the high-dBZ weather located? (xyz center of mass)
  3. Does a basic render work if we ignore the transfer function?
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr


def diagnose(nc_path: Path) -> None:
    ds = xr.open_dataset(nc_path, decode_times=False)
    dbz = np.asarray(ds["DBZ"].values, dtype=np.float32)
    if dbz.ndim == 4 and dbz.shape[0] == 1:
        dbz = dbz[0]

    # Handle the pyart fill value
    fill = ds["DBZ"].attrs.get("_FillValue")
    print(f"Gridded file _FillValue: {fill!r}")

    if fill is not None:
        dbz[dbz == fill] = np.nan

    finite = dbz[np.isfinite(dbz)]
    print(f"\nShape: {dbz.shape}")
    print(f"Total cells: {dbz.size:,}")
    print(f"Finite cells: {finite.size:,} ({100*finite.size/dbz.size:.2f}%)")
    if finite.size == 0:
        print("ALL CELLS ARE NaN — rendering is correct (nothing to render).")
        return

    print(f"\ndBZ distribution of finite cells:")
    bins = [-40, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80]
    hist, edges = np.histogram(finite, bins=bins)
    for i, c in enumerate(hist):
        pct = 100 * c / finite.size
        bar = "#" * int(pct * 0.6)
        print(f"  [{edges[i]:>+4.0f}, {edges[i+1]:>+4.0f}) : {c:>10,}  {pct:5.1f}%  {bar}")

    print(f"\nKey thresholds:")
    for t in (0, 10, 20, 30, 40, 50, 60):
        n = int((finite >= t).sum())
        pct = 100 * n / finite.size
        print(f"  >= {t:>2} dBZ : {n:>10,} cells ({pct:5.2f}% of valid)")

    # Where in the cube does high-dBZ weather live?
    print(f"\nLocation of weather (cells with dBZ >= 20):")
    mask = np.isfinite(dbz) & (dbz >= 20)
    if mask.any():
        z_idx, y_idx, x_idx = np.where(mask)
        z = np.asarray(ds["z"].values)
        y = np.asarray(ds["y"].values)
        x = np.asarray(ds["x"].values)
        print(f"  Z (altitude m): min={z[z_idx].min():.0f}  max={z[z_idx].max():.0f}  mean={z[z_idx].mean():.0f}")
        print(f"  Y (N-S m):      min={y[y_idx].min():.0f}  max={y[y_idx].max():.0f}  mean={y[y_idx].mean():.0f}")
        print(f"  X (E-W m):      min={x[x_idx].min():.0f}  max={x[x_idx].max():.0f}  mean={x[x_idx].mean():.0f}")
        print(f"  Cell count: {mask.sum():,}")
    else:
        print("  NO cells have dBZ >= 20 — this scan caught only very light precipitation or noise.")

    ds.close()


if __name__ == "__main__":
    nc = sys.argv[1] if len(sys.argv) > 1 else "output/RCTLS_09JUN2026_050131_L2B_STD_gridded.nc"
    diagnose(Path(nc))
