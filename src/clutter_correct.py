"""Clutter correction for polar DWR data.

The MOSDAC paper describes "a combination of a spatial continuity filter and a
fuzzy logic based echo classification algorithm." We implement the same idea in
two steps with dual-pol gating doing the heavy lifting:

  1. **Dual-pol gating (RHOHV)** — RHOHV (correlation coefficient) is near 1.0
     for pure rain, ~0.97 for mixed-phase, but drops below ~0.85 for biological
     targets (birds, insects) and hard non-weather targets (buildings, ground).
     Masking gates where RHOHV < threshold removes most ground clutter.

  2. **Spatial continuity filter** — isolated high-dBZ pixels surrounded by
     low-dBZ pixels are almost certainly clutter spikes (a single building
     showing as +95 dBZ in a sea of +20). A 2D median filter on each sweep
     suppresses these "salt" pixels without smoothing genuine storm cores.

Reference: Park et al. 2009 (the JPOLE fuzzy-logic classifier) — a full fuzzy
classifier uses ZDR/PHIDP/WIDTH as additional features. RHOHV alone is the
high-value 80% solution.
"""
from __future__ import annotations

import numpy as np
import pyart
from scipy.ndimage import median_filter


# ============================================================
#   Tunable knobs — try changing these and rerunning the pipeline.
# ============================================================
# RHOHV below this is considered non-meteorological.
#   0.80 = keep more (some clutter survives, light precip preserved)
#   0.85 = balanced  (MOSDAC-ish default)
#   0.90 = aggressive (drops melting-layer + light snow)
RHOHV_CLUTTER_THRESHOLD = 0.85

# Spatial median window (azimuth, range) applied per sweep.
# Larger = more aggressive spike suppression, but may eat real small cores.
MEDIAN_FILTER_SIZE = (3, 3)

# A pixel is flagged as a clutter spike if it is this many dBZ above the
# local median of its neighborhood.
SPIKE_THRESHOLD_DBZ = 15.0

# Hard upper bound: weather can't exceed ~75 dBZ. Anything above is residual
# clutter (RFI, hard targets that slipped past the RHOHV gate, etc.).
DBZ_PHYSICAL_MAX = 75.0
# ============================================================


def correct(radar: pyart.core.Radar) -> pyart.core.Radar:
    """Apply RHOHV gating + spatial spike filter to DBZ and VEL in-place."""
    dbz = np.ma.getdata(radar.fields["DBZ"]["data"]).astype(np.float32).copy()
    vel = np.ma.getdata(radar.fields["VEL"]["data"]).astype(np.float32).copy()
    rhohv = np.ma.getdata(radar.fields["RHOHV"]["data"]).astype(np.float32)

    # Preserve original mask state.
    dbz_mask = np.ma.getmaskarray(radar.fields["DBZ"]["data"]).copy()

    n_valid_before = (~dbz_mask).sum()

    # Step 1: RHOHV gate AND physical-maximum gate
    nonweather = rhohv < RHOHV_CLUTTER_THRESHOLD
    impossible = dbz > DBZ_PHYSICAL_MAX
    new_mask = dbz_mask | nonweather | impossible

    # Step 2: spatial continuity filter, applied sweep by sweep so we don't
    # average across sweep discontinuities.
    cleaned_dbz = dbz.copy()
    for sweep_slice in radar.iter_slice():
        sweep_dbz = dbz[sweep_slice].copy()
        sweep_mask = new_mask[sweep_slice]
        # Replace masked with sentinel for the median pass.
        filled = np.where(sweep_mask, -999.0, sweep_dbz)
        smoothed = median_filter(filled, size=MEDIAN_FILTER_SIZE, mode="nearest")
        spike = (sweep_dbz - smoothed) > SPIKE_THRESHOLD_DBZ
        new_mask[sweep_slice] |= spike
        cleaned_dbz[sweep_slice] = sweep_dbz

    # Write the corrected fields back.
    radar.fields["DBZ"]["data"] = np.ma.masked_array(cleaned_dbz, mask=new_mask)
    radar.fields["VEL"]["data"] = np.ma.masked_array(vel, mask=new_mask)

    n_valid_after = (~new_mask).sum()
    removed = n_valid_before - n_valid_after
    pct = 100.0 * removed / max(int(n_valid_before), 1)
    print(
        f"Clutter correction: masked {removed:,} cells "
        f"({pct:.1f}% of valid DBZ) "
        f"[RHOHV<{RHOHV_CLUTTER_THRESHOLD}, spike>{SPIKE_THRESHOLD_DBZ:.0f} dBZ]"
    )
    # Sanity check on remaining dBZ range — should now top out near 75 dBZ.
    remaining = cleaned_dbz[~new_mask]
    if remaining.size:
        print(f"  Cleaned DBZ range: [{remaining.min():.1f}, {remaining.max():.1f}] dBZ")

    return radar
