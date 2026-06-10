"""Significance filter: make our gridded cube match MOSDAC's L2C selectivity.

What we learned from validation:
  - The L2B file contains widespread weak echo (clear-air returns, drizzle,
    Bragg scatter, sea clutter) — the radar's clutter filter was OFF.
  - Our pipeline grids everything: ~1-3M valid cells per scan.
  - MOSDAC's official L2C keeps only ~5-8k cells: compact, significant storm
    cells. Their fuzzy-logic echo classifier rejects everything incoherent.

This filter approximates their selectivity with two steps:
  1. dBZ floor — drop cells below DBZ_FLOOR (their product's weak-echo cut).
  2. 3D connected-component despeckle — drop clusters smaller than
     MIN_CLUSTER_CELLS (kills scattered speckle, keeps storm cores; a real
     convective cell at this grid spacing spans hundreds of contiguous cells).

Output: <input>_official.nc with the same z/y/x layout, ready for any of our
renderers and for re-comparison against the official L2C.

Usage:
    python src/filter_official_style.py output/..._gridded.nc [floor] [min_cells]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr
from scipy import ndimage


DBZ_FLOOR = 10.0          # weak-echo cut (official product keeps almost nothing below this)
MIN_CLUSTER_CELLS = 300   # smallest 3D cluster to survive (1 km x 1 km x 250 m cells)


def official_filter(dbz: np.ndarray, floor: float, min_cells: int) -> np.ndarray:
    keep = np.isfinite(dbz) & (dbz >= floor)
    n_floor = int(keep.sum())

    # 26-connectivity: diagonal neighbours count as connected.
    labels, n_comp = ndimage.label(keep, structure=np.ones((3, 3, 3), dtype=bool))
    sizes = np.bincount(labels.ravel())
    big_labels = np.flatnonzero(sizes >= min_cells)
    big_labels = big_labels[big_labels != 0]  # 0 is background
    survives = np.isin(labels, big_labels)

    out = np.where(survives, dbz, np.nan).astype(np.float32)
    print(
        f"  floor {floor:.0f} dBZ: {n_floor:,} cells in {n_comp:,} clusters; "
        f"{len(big_labels)} clusters >= {min_cells} cells survive "
        f"-> {int(survives.sum()):,} cells"
    )
    return out


def main(nc_path: Path, floor: float, min_cells: int) -> Path:
    print(f"Loading {nc_path.name}...")
    ds = xr.open_dataset(nc_path, decode_times=False)
    dbz = np.asarray(ds["DBZ"].values, dtype=np.float32)
    squeeze_time = dbz.ndim == 4
    if squeeze_time:
        dbz = dbz[0]

    filtered = official_filter(dbz, floor, min_cells)

    out_data = filtered[np.newaxis] if squeeze_time else filtered
    ds_out = ds.copy()
    ds_out["DBZ"] = (ds["DBZ"].dims, out_data.astype(np.float32), ds["DBZ"].attrs)

    out_path = nc_path.with_name(nc_path.stem + "_official.nc")
    ds_out.to_netcdf(out_path)
    ds.close()
    print(f"Saved: {out_path}")
    return out_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python src/filter_official_style.py <gridded.nc> [floor] [min_cells]")
        sys.exit(1)
    nc = Path(sys.argv[1])
    floor = float(sys.argv[2]) if len(sys.argv) > 2 else DBZ_FLOOR
    min_cells = int(sys.argv[3]) if len(sys.argv) > 3 else MIN_CLUSTER_CELLS
    main(nc, floor, min_cells)
