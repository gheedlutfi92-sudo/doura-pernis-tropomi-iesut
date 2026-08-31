# IESUT: Industrial Emission Separation Using TROPOMI

Code accompanying the MSc dissertation *"Quantifying Industrial Atmospheric Emissions in Contrasting Regulatory Environments Using TROPOMI Sentinel-5P and Machine Learning: A Comparative Study of Doura Refinery, Baghdad, Iraq and Shell Pernis, Rotterdam, Netherlands"* (Ghid Albazrkan, MSc Geographical Information Systems, University of Aberdeen, 2026).

## Overview

IESUT trains a Random Forest regressor on TROPOMI tropospheric NO₂/SO₂ column pixels more than 12 km from a refinery (the "urban background"), then applies the model to pixels near the refinery. The residual between the observed column and the model's prediction is the industrial excess — the portion of the signal attributable to the refinery rather than general urban activity.

The method is applied and compared at two sites operating under very different regulatory conditions:
- **Doura Refinery, Baghdad, Iraq** — no binding emissions regulation
- **Shell Pernis, Rotterdam, Netherlands** — operating under the EU Emissions Trading System (EU ETS)

## Repository structure

```
gee/                Google Earth Engine JavaScript — TROPOMI + auxiliary data extraction
arcgis_toolbox/      ArcPy scripts for use inside ArcGIS Pro
python/              Standalone Python analysis scripts (run outside ArcGIS Pro)
```

| File | Purpose |
|---|---|
| `gee/gee_pixel_extraction_monthly_v2.js` | Extracts monthly TROPOMI NO₂/SO₂ and auxiliary variables (VIIRS, WorldPop, GHSL, SRTM, Landsat LST, ERA5) via Google Earth Engine |
| `arcgis_toolbox/iesut_arctool.py` | ArcPy script tool: fits the IESUT Random Forest model and computes industrial excess residuals inside ArcGIS Pro |
| `arcgis_toolbox/add_ehsa_fields.py` | Adds date/pixel-ID fields required for Emerging Hot Spot Analysis (EHSA) space-time cubes |
| `python/iesut.py` | Standalone IESUT model: Random Forest fitting, temporally blocked cross-validation, quantile regression forest uncertainty, sensitivity analysis |
| `python/tropomi_surface_conversion.py` | Converts TROPOMI column densities to surface concentrations (ERA5 PBLH-based) and compares against Al-Saidia ground station data |
| `python/rivm_pernis_comparison.py` | Ground validation of TROPOMI against the RIVM/DCMR Rotterdam monitoring network |
| `python/eutl_cross_validation.py` | Cross-validates annual mean IESUT NO₂ residuals at Pernis against EUTL-verified EU ETS NOx emissions, 2018–2024 (Section 5.5 of the dissertation) |
| `python/eu_ets_correlation.py` | Exploratory correlation of monthly EU ETS carbon (EUA) price against Pernis IESUT residuals; superseded by `eutl_cross_validation.py` above and not cited in the dissertation |
| `python/sen_slope_projection.py` | Mann-Kendall trend test and Sen's slope estimation; 10-year emissions projection |
| `python/wind_rose_both_sites.py` | Wind rose and NO₂ pollution rose generation for both sites |
| `python/md_to_docx.py` | Converts the dissertation Markdown draft to a formatted Word document |

## Data availability

Raw data are not redistributed in this repository. All satellite and reanalysis inputs are freely available from their original public sources:

- **TROPOMI Sentinel-5P** — [Google Earth Engine](https://developers.google.com/earth-engine/datasets/catalog/sentinel-5p) (`COPERNICUS/S5P/OFFL/L3/NO2`, `COPERNICUS/S5P/OFFL/L3/SO2`)
- **ERA5** — [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/)
- **WorldPop** — [worldpop.org](https://www.worldpop.org/)
- **VIIRS night-time lights** — [Google Earth Engine](https://developers.google.com/earth-engine/datasets/catalog/NOAA_VIIRS_DNB_MONTHLY_V1_VCMCFG)
- **RIVM/DCMR Rotterdam air quality data** — [Luchtmeetnet](https://www.luchtmeetnet.nl/)
- **EU ETS verified emissions (EUTL)** — [European Union Transaction Log](https://ec.europa.eu/clima/ets/)

Al-Saidia ground station data (Baghdad) were provided directly by the Iraqi Ministry of Environment and are not publicly redistributable; contact the Ministry directly for access.

## Requirements

Standalone Python scripts (`python/`): see `requirements.txt`.

ArcPy scripts (`arcgis_toolbox/`) require ArcGIS Pro (tested on 3.x) with its bundled Python environment — `arcpy` is not available via `pip`.

## Citation

If referencing this methodology, please cite the dissertation:

> Albazrkan, G. (2026) *Quantifying Industrial Atmospheric Emissions in Contrasting Regulatory Environments Using TROPOMI Sentinel-5P and Machine Learning: A Comparative Study of Doura Refinery, Baghdad, Iraq and Shell Pernis, Rotterdam, Netherlands*. MSc dissertation, University of Aberdeen.

## Generative AI Use

Generative AI (Claude, Anthropic) assisted with code development, debugging, and documentation for this repository, under the direction and review of the author. See the dissertation's Generative AI Declaration (Appendix C) for full details. All analytical decisions and interpretation of results are the author's own.

## License

MIT — see `LICENSE`.
