# repd-tools

Python tools for working with the UK **Renewable Energy Planning Database (REPD)**.

The REPD is published quarterly by DESNZ and tracks every renewable energy planning application in the UK from submission through to operational status or cancellation. 

This library converts the messier parts of the REPD dataset for a cleaner pandas dataframe usable for processing: date parsing, coordinate conversion from British National Grid (OSGB36) to WGS84, and clean export to GeoJSON.

**Download the data:** [gov.uk — REPD monthly extract](https://www.gov.uk/government/publications/renewable-energy-planning-database-monthly-extract)

---

## Install

```bash
pip install -e .
# or with uv
uv pip install -e .
```

Requires Python 3.11+.

---

## Quick start

```python
from datetime import datetime
from repd.processor import REPDProcessor

processor = REPDProcessor("REPD_Publication_Q4_2025.csv")

# Load, parse dates, convert OSGB36 → WGS84, return a GeoDataFrame
gdf = processor.process_pipeline(date=datetime(2025, 1, 1))

# Filter by outcome
cancelled = processor.filter_by_cancelled(gdf)
successful = processor.filter_by_successful(gdf)
```

See `examples/01_quickstart.ipynb` for a full walkthrough including maps and charts.

---

## What's included

| Module | Purpose |
|---|---|
| `repd.processor.REPDProcessor` | Load, filter, coordinate-convert, and export REPD data |
| `repd.constants` | Typed constants for all development status and technology type groups |
| `repd.visualise` | Matplotlib helpers for maps, outcome charts, delay distributions, and choropleths |

---

## Data files

The CSV and GeoJSON boundary files are included in the `./data` folder in this repo (the REPD CSV is ~4.5 MB and updated quarterly). Place them anywhere and pass the path to `REPDProcessor`:

```python
processor = REPDProcessor("path/to/REPD_Publication_Q4_2025.csv")
```

The `localauth.json` UK local authority boundary file is available from the [ONS Open Geography Portal](https://geoportal.statistics.gov.uk/).

---

## Coordinate system

The raw REPD data uses the British National Grid (OSGB36, EPSG:27700). This library reprojects all coordinates to WGS84 (EPSG:4326) so output GeoDataFrames and GeoJSON files work directly with standard mapping tools.

---

## Key fixes vs. the raw REPD columns

- **Date parsing** uses `dayfirst=True` — the REPD uses `dd/mm/yyyy` format, which pandas misparses by default.
- **Coordinate conversion** uses `always_xy=True` with pyproj to avoid axis-order ambiguity between OSGB36 and WGS84.
- **GeoJSON export** handles `NaT` / `NaN` values and serialises `Timestamp` columns to ISO strings.

---
## Contribution
Contribution to this repository is welcome. This is meant as a smaller data repository so smaller contributions and further data processing types are encouraged.

---

## License

[MIT](LICENSE) — © 2026 Damian Bemben
