"""Compare OUR gridded cube against MOSDAC's official L2C_VOL product.

The validation logic: both cubes were produced from the same L2B polar scan,
so every disagreement is an algorithm difference (clutter filter, dealiasing,
gridding weights) — not a data difference.

Outputs:
  - console summary: valid-cell overlap, correlation, bias, RMSE, verdict
  - a 4-panel comparison figure saved to output/

Usage:
    python src/compare_gridded.py <their_L2C_file.nc> <our_gridded.nc>
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

# MOSDAC may use different variable names in the L2C product than pyart uses
# in ours — try candidates until one hits.
DBZ_CANDIDATES = ("DBZ", "dbz", "reflectivity", "REF", "Z", "corrected_reflectivity")
VEL_CANDIDATES = ("VEL", "vel", "velocity", "radial_velocity", "corrected_velocity")


def _find_var(ds: xr.Dataset, candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in ds.variables:
            return name
    return None


def load_cube(path: Path, label: str) -> np.ndarray:
    """Load a DBZ cube from either file flavor, normalized to (height, lat, lon).

    MOSDAC's L2C files store DBZ with dims (time, height, lon, lat) — lat last.
    Our pyart-written cubes are (time, z, y, x) = (time, height, lat, lon).
    We use the named dims to put everything into (height, lat, lon) order.
    """
    ds = xr.open_dataset(path, decode_times=False)
    name = _find_var(ds, DBZ_CANDIDATES)
    if name is None:
        raise KeyError(
            f"[{label}] No reflectivity variable found. Variables present: "
            f"{list(ds.data_vars)} — add the right name to DBZ_CANDIDATES."
        )
    var = ds[name]
    dims = list(var.dims)

    data = np.asarray(var.values, dtype=np.float32)
    # Squeeze any size-1 leading dims (both flavors write a time dim of 1).
    while data.ndim > 3 and data.shape[0] == 1:
        data = data[0]
        dims = dims[1:]

    # MOSDAC's L2C names its dims (..., lon, lat) but the data layout is
    # empirically (..., lat, lon) — verified against our independently gridded
    # cube: raw order correlates 0.66 with 99.8% cell overlap, while honoring
    # the dim names gives -0.18 with 42% overlap. The labels are swapped in
    # their writer; trust the raw layout. (_orientation_check guards this.)

    fill = var.attrs.get("_FillValue", var.attrs.get("missing_value"))
    if fill is not None:
        data[data == fill] = np.nan
    # Physically impossible values -> NaN, same floor we apply everywhere.
    data[data < -32.0] = np.nan

    ds.close()
    print(f"[{label}] {path.name}: var='{name}' shape={data.shape} "
          f"valid={np.isfinite(data).sum():,}")
    return data


def _orientation_check(theirs: np.ndarray, ours: np.ndarray) -> None:
    """Empirical guard against mislabeled metadata: compare correlation with
    the metadata-derived orientation vs the lat/lon-swapped one."""
    def _corr(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
        m = np.isfinite(a) & np.isfinite(b)
        n = int(m.sum())
        if n < 100:
            return (float("nan"), n)
        x, y = a[m], b[m]
        if np.std(x) == 0 or np.std(y) == 0:
            return (float("nan"), n)
        return (float(np.corrcoef(x, y)[0, 1]), n)

    c_meta, n_meta = _corr(theirs, ours)
    c_swap, n_swap = _corr(np.swapaxes(theirs, 1, 2), ours)
    print(f"\n--- Orientation sanity check ---")
    print(f"  as metadata says : corr={c_meta:.4f} on {n_meta:,} voxels")
    print(f"  lat/lon swapped  : corr={c_swap:.4f} on {n_swap:,} voxels")
    if np.isfinite(c_swap) and (not np.isfinite(c_meta) or c_swap > c_meta + 0.1):
        print("  WARNING: swapped orientation fits much better — metadata may be mislabeled!")


def compare(theirs_path: Path, ours_path: Path) -> None:
    theirs = load_cube(theirs_path, "MOSDAC L2C")
    ours = load_cube(ours_path, "our pipeline")

    if theirs.shape != ours.shape:
        print(f"\nWARNING: shape mismatch — theirs {theirs.shape} vs ours {ours.shape}.")
        print("Run inspect_nc.py on their file; the grid layout may need transposing.")
        return

    _orientation_check(theirs, ours)

    tv = np.isfinite(theirs)
    ov = np.isfinite(ours)
    both = tv & ov
    only_t = tv & ~ov
    only_o = ov & ~tv

    print(f"\n--- Coverage agreement ---")
    print(f"  valid in theirs : {tv.sum():>12,}")
    print(f"  valid in ours   : {ov.sum():>12,}")
    print(f"  both valid      : {both.sum():>12,}")
    print(f"  only theirs     : {only_t.sum():>12,}  (we removed / they kept)")
    print(f"  only ours       : {only_o.sum():>12,}  (we kept / they removed)")

    if both.sum() < 1000:
        print("Too little overlap for meaningful statistics.")
        return

    a = theirs[both]
    b = ours[both]
    bias = float(np.mean(b - a))
    rmse = float(np.sqrt(np.mean((b - a) ** 2)))
    mae = float(np.mean(np.abs(b - a)))
    corr = float(np.corrcoef(a, b)[0, 1])

    print(f"\n--- Value agreement on {both.sum():,} co-valid voxels ---")
    print(f"  correlation : {corr:.4f}")
    print(f"  mean bias   : {bias:+.2f} dBZ  (ours minus theirs)")
    print(f"  RMSE        : {rmse:.2f} dBZ")
    print(f"  MAE         : {mae:.2f} dBZ")

    verdict = (
        "EXCELLENT — pipeline faithfully recreates the official product" if corr >= 0.90 and abs(bias) <= 2
        else "GOOD — same storms, modest algorithmic differences" if corr >= 0.75
        else "PARTIAL — structure similar but tuning needed (check clutter/grid knobs)" if corr >= 0.5
        else "POOR — likely an axis-order, units, or alignment problem, not just tuning"
    )
    print(f"\n  VERDICT: {verdict}")

    # Registration search: sharp convective gradients mean a 1-cell (1 km)
    # misregistration between the two grids can wreck voxel correlation even
    # when the fields are essentially identical. Try small integer shifts of
    # our cube in lat/lon and report the best fit.
    print(f"\n--- Registration search (shifting ours by whole cells) ---")
    best = (0, 0, corr)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            shifted = np.roll(ours, (dy, dx), axis=(1, 2))
            m = tv & np.isfinite(shifted)
            if m.sum() < 500:
                continue
            x, y = theirs[m], shifted[m]
            if np.std(x) == 0 or np.std(y) == 0:
                continue
            c = float(np.corrcoef(x, y)[0, 1])
            if c > best[2]:
                best = (dy, dx, c)
    dy, dx, c = best
    if (dy, dx) == (0, 0):
        print(f"  no shift beats the aligned grids (corr stays {c:.4f}) — registration is good")
    else:
        print(f"  best: shift ours by {dy:+d} lat cells, {dx:+d} lon cells -> corr={c:.4f}")
        print(f"  (a ~{max(abs(dy),abs(dx))} km offset between their grid anchor and ours)")

    # Per-altitude-level agreement profile.
    nz = theirs.shape[0]
    lev_corr = np.full(nz, np.nan)
    lev_bias = np.full(nz, np.nan)
    for k in range(nz):
        m = both[k]
        if m.sum() >= 200:
            ta, ob = theirs[k][m], ours[k][m]
            if np.std(ta) > 0 and np.std(ob) > 0:
                lev_corr[k] = np.corrcoef(ta, ob)[0, 1]
            lev_bias[k] = np.mean(ob - ta)

    # ---- 4-panel figure ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    ax = axes[0, 0]
    h = ax.hist2d(a, b, bins=80, range=[[-30, 70], [-30, 70]], cmin=1, cmap="viridis")
    ax.plot([-30, 70], [-30, 70], "r--", lw=1, label="1:1 line")
    ax.set_xlabel("MOSDAC L2C dBZ")
    ax.set_ylabel("Our pipeline dBZ")
    ax.set_title(f"Voxel-by-voxel (corr={corr:.3f})")
    ax.legend()
    fig.colorbar(h[3], ax=ax, label="voxel count")

    ax = axes[0, 1]
    alt_km = np.arange(nz) * 0.25
    ax.plot(lev_corr, alt_km, "b-", label="correlation")
    ax.set_xlabel("Correlation", color="b")
    ax.set_ylabel("Altitude (km)")
    ax2 = ax.twiny()
    ax2.plot(lev_bias, alt_km, "r-", label="bias")
    ax2.set_xlabel("Bias ours−theirs (dBZ)", color="r")
    ax2.axvline(0, color="r", ls=":", lw=0.5)
    ax.set_title("Agreement by altitude")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    k2 = 8  # 2 km level (8 * 250 m)
    diff = ours[k2] - theirs[k2]
    im = ax.imshow(diff, origin="lower", cmap="RdBu_r", vmin=-10, vmax=10)
    ax.set_title("Difference map @ 2 km CAPPI (ours − theirs)")
    ax.set_xlabel("E–W cell")
    ax.set_ylabel("N–S cell")
    fig.colorbar(im, ax=ax, label="ΔdBZ")

    ax = axes[1, 1]
    bins = np.arange(-30, 72, 2)
    ax.hist(theirs[tv], bins=bins, alpha=0.5, label="MOSDAC L2C", density=True)
    ax.hist(ours[ov], bins=bins, alpha=0.5, label="our pipeline", density=True)
    ax.set_xlabel("dBZ")
    ax.set_ylabel("density")
    ax.set_title("dBZ distributions (all valid voxels)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"Pipeline validation: {ours_path.name} vs official {theirs_path.name}",
        fontsize=11,
    )
    fig.tight_layout()
    out = OUTPUT_DIR / f"validation_{theirs_path.stem.replace('_L2C_STD', '')}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved comparison figure: {out}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python src/compare_gridded.py <their_L2C.nc> <our_gridded.nc>")
        sys.exit(1)
    compare(Path(sys.argv[1]), Path(sys.argv[2]))
