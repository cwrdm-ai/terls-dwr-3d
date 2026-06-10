"""Inspect a MOSDAC TERLS DWR NetCDF file.

Prints every dimension, variable, attribute, and a small data sample so we know
exactly what schema MOSDAC uses before we touch the renderer.

Usage:
    python src/inspect_nc.py data/your_file.nc
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr


def inspect(path: Path) -> None:
    print(f"\n=== File: {path.name}  ({path.stat().st_size / 1e6:.1f} MB) ===\n")
    ds = xr.open_dataset(path, decode_times=False)

    print("--- Dimensions ---")
    for name, size in ds.sizes.items():
        print(f"  {name:20s} = {size}")

    print("\n--- Coordinates ---")
    for name, coord in ds.coords.items():
        vals = coord.values
        if vals.ndim == 1 and vals.size > 0:
            print(f"  {name:20s} shape={vals.shape} range=[{vals.min():.4f}, {vals.max():.4f}] dtype={vals.dtype}")
        else:
            print(f"  {name:20s} shape={vals.shape} dtype={vals.dtype}")

    print("\n--- Data variables ---")
    for name, var in ds.data_vars.items():
        attrs = var.attrs
        units = attrs.get("units", "?")
        long_name = attrs.get("long_name", attrs.get("standard_name", ""))
        fill = attrs.get("_FillValue", attrs.get("missing_value", None))
        v = var.values

        # Numeric vs string handling — string metadata (sweep_mode etc.) has no range.
        if np.issubdtype(v.dtype, np.number):
            finite = np.isfinite(v) if np.issubdtype(v.dtype, np.floating) else np.ones_like(v, dtype=bool)
            if fill is not None and np.issubdtype(v.dtype, np.floating):
                finite &= (v != fill)
            valid = v[finite] if finite.any() else np.array([])
            rng = f"[{valid.min():.2f}, {valid.max():.2f}]" if valid.size else "all-missing"
        else:
            sample = v.flatten()[:3]
            rng = f"sample={list(sample)}"

        print(f"  {name:25s} shape={var.shape}  dtype={var.dtype}  units={units}  range={rng}")
        if long_name:
            print(f"    long_name: {long_name}")
        if fill is not None:
            print(f"    fill_value: {fill}")

    print("\n--- Global attributes ---")
    for k, v in ds.attrs.items():
        s = str(v)
        if len(s) > 100:
            s = s[:97] + "..."
        print(f"  {k}: {s}")

    ds.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python src/inspect_nc.py <path-to-nc-file>")
        sys.exit(1)
    inspect(Path(sys.argv[1]))
