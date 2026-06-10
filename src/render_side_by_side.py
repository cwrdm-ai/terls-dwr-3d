"""The decisive validation visual: official L2C vs our pipeline, same cells.

Left  : MOSDAC's official L2C product (their values, their cells).
Right : OUR pipeline's values sampled at exactly their cells.

If the two panels look identical, the pipeline reproduces the official
product wherever the official product has data — the only difference
between the full products is their classifier's selectivity.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import Normalize

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def main(stamp: str) -> Path:
    ds = xr.open_dataset(
        f"data/Jun26_182954/RCTLS_06JUN2026_{stamp}_L2C_STD.nc", decode_times=False
    )
    theirs = ds["DBZ"].values[0]          # raw layout = (height, lat, lon)
    h_km = ds["height"].values / 1000.0
    lat_ax = ds["latitude"].values
    lon_ax = ds["longitude"].values
    ds.close()

    ds = xr.open_dataset(
        f"output/RCTLS_06JUN2026_{stamp}_L2B_STD_gridded.nc", decode_times=False
    )
    ours = ds["DBZ"].values[0]            # (z, y=lat, x=lon), registration verified exact
    ds.close()

    mask = np.isfinite(theirs)
    ours_at_their_cells = np.where(mask, ours, np.nan)

    m = mask & np.isfinite(ours)
    corr = float(np.corrcoef(theirs[m], ours[m])[0, 1])
    bias = float(np.mean(ours[m] - theirs[m]))

    H, LAT, LON = np.meshgrid(h_km, lat_ax, lon_ax, indexing="ij")
    norm = Normalize(vmin=-25, vmax=50)

    fig = plt.figure(figsize=(14, 6))
    panels = [
        (theirs, "MOSDAC official L2C product"),
        (ours_at_their_cells, "OUR pipeline — values at the same cells"),
    ]
    for i, (data, title) in enumerate(panels):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        keep = np.isfinite(data)
        sc = ax.scatter(
            LON[keep], LAT[keep], H[keep], c=data[keep],
            cmap="jet", norm=norm, s=4, marker="s", alpha=0.6, linewidths=0,
        )
        ax.set_xlabel("Longitude (°E)", labelpad=8)
        ax.set_ylabel("Latitude (°N)", labelpad=8)
        ax.set_zlabel("Altitude (km)", labelpad=6)
        ax.set_zlim(0, 21)
        ax.set_zticks([0, 7, 14, 21])
        ax.set_title(title, fontsize=11)
        ax.view_init(elev=15, azim=-60)
        cb = fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.10)
        cb.set_label("Reflectivity (dBZ)")

    fig.suptitle(
        f"TERLS DWR 06 Jun 2026 {stamp[:2]}:{stamp[2:4]}:{stamp[4:]} UTC — "
        f"corr={corr:.2f}, bias={bias:+.1f} dBZ on {int(m.sum()):,} shared cells",
        fontsize=12,
    )
    out = OUTPUT_DIR / f"side_by_side_{stamp}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "140341")
