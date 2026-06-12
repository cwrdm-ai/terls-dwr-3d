"""MOSDAC-documented clutter correction (v2) — the literal cited algorithms.

The TERLS DWR Product Readme + product page name the exact method:
  "The radar reflectivity was corrected using a combination of a spatial
   continuity filter and a fuzzy logic based-echo classification algorithm.
   ... applied on the PPI level data at each elevation angle"
with references to Gabella & Notarpietro (2002) and Vulpiani et al. (2012) —
both implemented verbatim in wradlib (the library MOSDAC themselves used):

  1. wradlib.classify.filter_gabella   — echo continuity + minimum echo area
  2. wradlib.classify.classify_echo_fuzzy — fuzzy combination of dual-pol
     texture features (ZDR, RHOHV, PHIDP textures + Doppler velocity)

Both run per elevation sweep (PPI level), exactly as documented.
"""
from __future__ import annotations

import numpy as np
import pyart
import wradlib.dp as wrl_dp
from wradlib.classify import classify_echo_fuzzy, filter_gabella


# Below this probability of being meteorological, a gate is called non-met.
# 0.5 is wradlib's conventional decision threshold for the Vulpiani classifier.
MET_PROB_THRESHOLD = 0.5

# Physical ceiling — weather cannot exceed ~75 dBZ (kept from v1).
DBZ_PHYSICAL_MAX = 75.0


def _sweep_view(radar: pyart.core.Radar, field: str, sl: slice) -> np.ndarray:
    """Raw float view of one sweep (n_az x n_gates) with NaN where masked."""
    data = radar.fields[field]["data"][sl]
    raw = np.ma.getdata(data).astype(np.float64)
    raw[np.ma.getmaskarray(data)] = np.nan
    return raw


def correct(radar: pyart.core.Radar,
            met_prob_threshold: float = MET_PROB_THRESHOLD,
            nan_policy: str = "kill",
            rescue_dbz: float = 10.0) -> pyart.core.Radar:
    """Apply Gabella + Vulpiani fuzzy QC per sweep to DBZ and VEL in-place.

    Parameters open for calibration against the official L2C product
    (MOSDAC's exact constants live in unpublished SAC reports):

    met_prob_threshold : gates with P(meteorological) below this are non-met.
    nan_policy : what to do where the classifier has no dual-pol signal:
        'kill'   — treat as non-met (harshest; v2 default)
        'rescue' — keep if DBZ >= rescue_dbz (strong echo earns trust)
        'keep'   — leave to the Gabella filter alone
    rescue_dbz : threshold for the 'rescue' policy.
    """
    dbz_full = np.ma.getdata(radar.fields["DBZ"]["data"]).astype(np.float32).copy()
    vel_full = np.ma.getdata(radar.fields["VEL"]["data"]).astype(np.float32).copy()
    base_mask = np.ma.getmaskarray(radar.fields["DBZ"]["data"]).copy()
    n_valid_before = (~base_mask).sum()

    new_mask = base_mask.copy()
    new_mask |= dbz_full > DBZ_PHYSICAL_MAX

    n_gab = n_fuzzy = 0
    for i, sl in enumerate(radar.iter_slice()):
        dbz = _sweep_view(radar, "DBZ", sl)
        vel = _sweep_view(radar, "VEL", sl)
        zdr = _sweep_view(radar, "ZDR", sl)
        rho = _sweep_view(radar, "RHOHV", sl)
        phi = _sweep_view(radar, "PHIDP", sl)

        # --- Step 1: Vulpiani 2012 fuzzy echo classification ---
        # Decision variables are TEXTURES of the dual-pol moments (local
        # roughness — meteorological echo is smooth, clutter is noisy),
        # plus Doppler velocity (clutter sits near zero) and a static
        # clutter map (none available for TERLS -> zeros).
        dat = {
            "zdr": wrl_dp.texture(zdr),
            "rho": wrl_dp.texture(rho),
            "phi": wrl_dp.texture(phi),
            "dop": vel,
            "map": np.zeros_like(dbz, dtype=bool),
            "rho2": rho,
        }
        prob_met, nan_mask = classify_echo_fuzzy(dat)
        # classify_echo_fuzzy returns MASKED arrays (masked where the radar
        # recorded no usable dual-pol signal). Convert to plain ndarrays so
        # |= and .sum() see every element.
        prob = np.ma.filled(prob_met, np.nan)
        nm = np.ma.filled(nan_mask, True).astype(bool)
        nm |= ~np.isfinite(prob)

        # Where the classification is VALID, trust the fuzzy probability.
        nonmet_dualpol = ~nm & (np.where(np.isfinite(prob), prob, 0.0) < met_prob_threshold)
        # Where it is NOT valid, apply the chosen policy.
        if nan_policy == "kill":
            nonmet_nan = nm
        elif nan_policy == "rescue":
            nonmet_nan = nm & ~(np.where(np.isfinite(dbz), dbz, -99.0) >= rescue_dbz)
        elif nan_policy == "keep":
            nonmet_nan = np.zeros_like(nm)
        else:
            raise ValueError(f"unknown nan_policy {nan_policy!r}")
        nonmet = np.asarray(nonmet_dualpol | nonmet_nan, dtype=bool)
        n_fuzzy += int((nonmet & np.isfinite(dbz)).sum())

        # --- Step 2: Gabella 2002 spatial continuity filter ---
        # Echo continuity + minimum echo area on the reflectivity sweep.
        clutter = filter_gabella(
            np.where(np.isfinite(dbz), dbz, -np.inf),
            wsize=5, thrsnorain=0.0, tr1=6.0, n_p=8, tr2=1.3, rm_nans=False,
        )
        n_gab += int((clutter & np.isfinite(dbz) & ~nonmet).sum())

        new_mask[sl] |= nonmet | clutter

    radar.fields["DBZ"]["data"] = np.ma.masked_array(dbz_full, mask=new_mask)
    radar.fields["VEL"]["data"] = np.ma.masked_array(vel_full, mask=new_mask)

    n_valid_after = (~new_mask).sum()
    removed = int(n_valid_before - n_valid_after)
    pct = 100.0 * removed / max(int(n_valid_before), 1)
    print(
        f"QC v2 (documented): masked {removed:,} gates ({pct:.1f}%) — "
        f"fuzzy(Vulpiani) flagged {n_fuzzy:,}, Gabella added {n_gab:,}"
    )
    remaining = dbz_full[~new_mask]
    if remaining.size:
        print(f"  Cleaned DBZ range: [{remaining.min():.1f}, {remaining.max():.1f}] dBZ, "
              f"{remaining.size:,} gates survive")
    return radar
