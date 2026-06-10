# TERLS DWR 3D Visualization Pipeline

Recreates ISRO/MOSDAC's **3D Volumetric TERLS DWR product** from raw polar radar
files, and renders it in the IMD/MOSDAC presentation styles.

TERLS = Thumba Equatorial Rocket Launching Station (8.537°N, 76.866°E, Kerala,
India), home of a C-band dual-polarization Doppler Weather Radar operated by
ISRO. Data: [MOSDAC](https://www.mosdac.gov.in) (free registration required).

## Pipeline

```
L2B polar NetCDF (11 sweeps × 360 az × 1600 gates)
  │  io_radar.py          – CfRadial reader + MOSDAC quirk fixes
  │  clutter_correct.py   – RHOHV gate + spatial spike filter + 75 dBZ cap
  │  dealias_velocity.py  – region-based velocity dealiasing (pyart)
  │  grid_to_cartesian.py – polar → 81×481×481 cube (1 km × 1 km × 250 m)
  ▼
gridded cube  ──► render_dwr.py            (PyVista volume render, dark cinematic)
              ──► render_imd_exact.py      (matplotlib 3D scatter, IMD operational style)
              ──► render_stacked_cappi.py  (stacked CAPPI sheets, dark IMD style)
```

Run end-to-end:

```bash
python src/pipeline.py data/RCTLS_<date>_<time>_L2B_STD.nc
```

Then render any style from the cube in `output/`.

## Validation against the official product

Our cubes were validated against MOSDAC's official `RCTLS_L2C_VOL` gridded
product on 4 matched scan pairs (06 Jun 2026): **99–100 % of the official
product's cells are present in ours, mean bias −0.5 to −2.1 dBZ, voxel
correlation 0.62–0.75, grid registration exact** (`src/compare_gridded.py`).
The official product additionally applies a fuzzy-logic echo classifier that
keeps only ~5–8 k precipitating-storm cells per scan; our cubes retain
everything the radar saw (1–3 M cells).

## MOSDAC file quirks handled by this code

| File | Quirk | Fix |
|---|---|---|
| L2B | `_FillValue = 0` on DBZ (0 dBZ is real data) | re-mask only < −32 dBZ |
| L2B | `time:units` is the literal string `"yyyy-mm-ddThh:mm:ssZ"` | parse scan time from filename |
| L2B | Nyquist velocity missing | inject ±24 m/s (TERLS dual-PRF) |
| L2C | `lon`/`lat` dimension names swapped vs actual data layout | trust raw order (verified empirically) |

## Requirements

Python 3.11+, `arm-pyart`, `pyvista`, `xarray`, `netCDF4`, `scipy`, `matplotlib`.

## Acknowledgements

- Data: MOSDAC / Space Applications Centre, ISRO
- Py-ART: Helmus & Collis (2016), JORS, doi:10.5334/jors.119
- pyiwr toolkit (ISRO weather radar Python library) for format reference
