"""Thin wrapper around pyart.io.read_cfradial that handles MOSDAC's quirks.

Three fixes we apply at load:

1. **Override the bogus _FillValue=0 on DBZ.** MOSDAC declared 0 dBZ as the
   missing-value sentinel, but 0 dBZ is a *real* measurement (very light
   drizzle). We unmask DBZ and keep only physically impossible values
   (< -32 dBZ) as missing.

2. **Set Nyquist velocity explicitly.** MOSDAC's file omits the
   instrument_parameters/nyquist_velocity field. From the VEL data range
   (±23.8 m/s) we infer the radar's Nyquist is ~24 m/s. The dealiasing
   algorithm refuses to run without this number.

3. **Fix the malformed time units string.** MOSDAC's file writes the literal
   CF template `"yyyy-mm-ddThh:mm:ssZ"` instead of filling in the scan time.
   pyart's gridder calls cftime to parse this and crashes. We parse the
   timestamp out of the filename (e.g. RCTLS_09JUN2026_050131_...) and set
   the proper units string.
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pyart


NYQUIST_MS = 24.0   # inferred from TERLS dual-PRF VEL range

# Matches RCTLS_DDMMMYYYY_HHMMSS_..., e.g. RCTLS_09JUN2026_050131_L2B_STD.nc
_FILENAME_TIME_RE = re.compile(
    r"^[A-Z]+_(\d{2})([A-Z]{3})(\d{4})_(\d{2})(\d{2})(\d{2})_"
)


def _parse_timestamp_from_filename(filename: str) -> datetime | None:
    m = _FILENAME_TIME_RE.match(filename)
    if not m:
        return None
    day, mon, year, hh, mm, ss = m.groups()
    try:
        return datetime.strptime(f"{day}{mon}{year} {hh}{mm}{ss}", "%d%b%Y %H%M%S")
    except ValueError:
        return None


def read(path: str | Path) -> pyart.core.Radar:
    path = Path(path)
    radar = pyart.io.read_cfradial(str(path))

    # Fix 1: un-mask DBZ values incorrectly flagged by _FillValue=0.
    if "DBZ" in radar.fields:
        dbz = radar.fields["DBZ"]["data"]
        raw = np.ma.getdata(dbz).astype(np.float32)
        mask = (raw < -32.0) | ~np.isfinite(raw)
        radar.fields["DBZ"]["data"] = np.ma.masked_array(raw, mask=mask)

    # Fix 2: inject Nyquist velocity if missing.
    if radar.instrument_parameters is None:
        radar.instrument_parameters = {}
    if "nyquist_velocity" not in radar.instrument_parameters:
        nyq = np.full(radar.nrays, NYQUIST_MS, dtype=np.float32)
        radar.instrument_parameters["nyquist_velocity"] = {
            "data": nyq,
            "units": "meters_per_second",
            "long_name": "Nyquist velocity",
            "comments": "Inferred from VEL data range; not present in source file.",
        }

    # Fix 3: rewrite the time units string. The file's value is the literal
    # template "yyyy-mm-ddThh:mm:ssZ" which cftime cannot parse.
    units = radar.time.get("units", "")
    if "yyyy" in units.lower() or not units.lower().startswith("seconds since"):
        scan_dt = _parse_timestamp_from_filename(path.name)
        if scan_dt is None:
            raise ValueError(
                f"Cannot fix malformed time units '{units}': filename "
                f"'{path.name}' doesn't match RCTLS_DDMMMYYYY_HHMMSS_ pattern."
            )
        radar.time["units"] = scan_dt.strftime("seconds since %Y-%m-%d %H:%M:%S")
        print(f"  Fixed malformed time units -> '{radar.time['units']}'")

    # Fix 4: rebuild broken sweep ray indices. The file writes ZEROS for both
    # sweep_start_ray_index and sweep_end_ray_index, which makes pyart's
    # iter_slice() return 1-ray "sweeps" — silently crippling every per-sweep
    # algorithm (clutter filters, dealiasing). Rays are stored contiguously,
    # uniform count per sweep, so the indices are reconstructable.
    ssri = np.asarray(radar.sweep_start_ray_index["data"])
    seri = np.asarray(radar.sweep_end_ray_index["data"])
    if radar.nsweeps > 1 and ssri.ptp() == 0 and seri.ptp() == 0:
        rays_per_sweep = radar.nrays // radar.nsweeps
        starts = np.arange(radar.nsweeps, dtype=np.int32) * rays_per_sweep
        radar.sweep_start_ray_index["data"] = starts
        radar.sweep_end_ray_index["data"] = starts + rays_per_sweep - 1
        print(f"  Fixed broken sweep ray indices -> {rays_per_sweep} rays/sweep")

    return radar
