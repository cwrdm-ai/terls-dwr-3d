"""Calibrate the documented QC pipeline against the official L2C product.

MOSDAC's exact tuning constants (fuzzy threshold, no-signal-gate policy,
gridding radius) live in unpublished SAC reports. But we hold 4 matched
(L2B input, official L2C output) pairs — enough to FIT those constants by
maximizing agreement with their product.

Stage 1: sweep (threshold x nan_policy x rescue_dbz) on the 05:14 pair,
         grid DBZ in memory with constant 700 m RoI, score each config.
Stage 2: take the best config (by Jaccard) and confirm on all 4 pairs.

Scoring vs the official cube:
  jaccard   = both / (theirs + ours - both)     <- headline metric
  retention = both / theirs                     (their cells we keep)
  corr/bias on co-valid voxels
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pyart
import xarray as xr

import io_radar
import clutter_correct_v2 as qc

DATA = Path(__file__).resolve().parent.parent / "data" / "Jun26_182954"
STAMPS = ["051420", "000817", "100523", "140341"]   # 05:14 first (fit pair)

N_ALT, N_LAT, N_LON = 81, 481, 481
ALT_MAX, HALF = 20_000.0, 240_000.0


def load_official(stamp: str) -> np.ndarray:
    ds = xr.open_dataset(DATA / f"RCTLS_06JUN2026_{stamp}_L2C_STD.nc", decode_times=False)
    t = np.asarray(ds["DBZ"].values[0], dtype=np.float32)   # raw layout (h, lat, lon)
    ds.close()
    t[t < -32.0] = np.nan
    return t


def grid_dbz(radar: pyart.core.Radar, roi_m: float) -> np.ndarray:
    g = pyart.map.grid_from_radars(
        (radar,),
        grid_shape=(N_ALT, N_LAT, N_LON),
        grid_limits=((0.0, ALT_MAX), (-HALF, HALF), (-HALF, HALF)),
        fields=["DBZ"],
        weighting_function="Barnes2",
        roi_func="constant",
        constant_roi=roi_m,
    )
    d = g.fields["DBZ"]["data"][0] if g.fields["DBZ"]["data"].ndim == 4 else g.fields["DBZ"]["data"]
    arr = np.ma.filled(d, np.nan).astype(np.float32)
    return arr


def score(theirs: np.ndarray, ours: np.ndarray) -> dict:
    tv, ov = np.isfinite(theirs), np.isfinite(ours)
    both = tv & ov
    nb = int(both.sum())
    out = {
        "theirs": int(tv.sum()), "ours": int(ov.sum()), "both": nb,
        "jaccard": nb / max(int((tv | ov).sum()), 1),
        "retention": nb / max(int(tv.sum()), 1),
        "corr": np.nan, "bias": np.nan,
    }
    if nb >= 200:
        a, b = theirs[both], ours[both]
        if np.std(a) > 0 and np.std(b) > 0:
            out["corr"] = float(np.corrcoef(a, b)[0, 1])
        out["bias"] = float(np.mean(b - a))
    return out


def snapshot(radar):
    return {f: radar.fields[f]["data"].copy() for f in ("DBZ", "VEL")}


def restore(radar, snap):
    for f, d in snap.items():
        radar.fields[f]["data"] = d.copy()


def fmt(cfg, s) -> str:
    return (f"  thr={cfg[0]:.1f} policy={cfg[1]:<6s} rescue={str(cfg[2]):>4s} roi={cfg[3]:>4.0f}: "
            f"ours={s['ours']:>9,} both={s['both']:>6,} "
            f"J={s['jaccard']:.3f} ret={s['retention']:.3f} "
            f"corr={s['corr']:.3f} bias={s['bias']:+.2f}")


def main() -> None:
    t0 = time.time()
    stamp = STAMPS[0]
    print(f"=== Stage 1: fit on {stamp} ===")
    theirs = load_official(stamp)
    radar = io_radar.read(DATA / f"RCTLS_06JUN2026_{stamp}_L2B_STD.nc")
    snap = snapshot(radar)

    # (threshold, nan_policy, rescue_dbz, roi_m)
    configs = [
        (0.5, "kill",   None, 700.0),
        (0.3, "kill",   None, 700.0),
        (0.5, "rescue", 10.0, 700.0),
        (0.5, "rescue", 20.0, 700.0),
        (0.3, "rescue", 10.0, 700.0),
        (0.3, "rescue", 20.0, 700.0),
        (0.5, "keep",   None, 700.0),
        (0.5, "rescue", 10.0, 1000.0),
        (0.5, "rescue", 10.0, 500.0),
    ]

    results = []
    for cfg in configs:
        thr, pol, resc, roi = cfg
        restore(radar, snap)
        qc.correct(radar, met_prob_threshold=thr, nan_policy=pol,
                   rescue_dbz=resc if resc is not None else 10.0)
        ours = grid_dbz(radar, roi)
        s = score(theirs, ours)
        results.append((cfg, s))
        print(fmt(cfg, s), flush=True)

    best_cfg, best_s = max(results, key=lambda r: r[1]["jaccard"])
    print(f"\nBest by Jaccard: thr={best_cfg[0]} policy={best_cfg[1]} "
          f"rescue={best_cfg[2]} roi={best_cfg[3]:.0f} (J={best_s['jaccard']:.3f})")

    print(f"\n=== Stage 2: confirm best config on all pairs ===")
    for st in STAMPS:
        if st == stamp:
            print(f"  {st}: " + fmt(best_cfg, best_s).strip())
            continue
        theirs_i = load_official(st)
        radar_i = io_radar.read(DATA / f"RCTLS_06JUN2026_{st}_L2B_STD.nc")
        qc.correct(radar_i, met_prob_threshold=best_cfg[0], nan_policy=best_cfg[1],
                   rescue_dbz=best_cfg[2] if best_cfg[2] is not None else 10.0)
        ours_i = grid_dbz(radar_i, best_cfg[3])
        s_i = score(theirs_i, ours_i)
        print(f"  {st}: " + fmt(best_cfg, s_i).strip(), flush=True)

    print(f"\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
