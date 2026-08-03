# traffic_interpolation

Regressing and interpolating vehicular activity data across road networks to produce high-resolution inputs for the VEIN emissions model.

**Status:** Code and models for the XGBoost paper submission (2026).

The **R and Python pipelines are mirror implementations**: given the same inputs they produce statistically equivalent models and figures (R² differences ≲ 0.02, typically < 0.01).

---

## Structure

```
traffic_interpolation/
├── data/                        # Ignored — populated by download_data.sh
├── experiments/
│   └── 2026_paper_xgboost_R/
│       ├── run_pipeline.R       # Runs all R steps 00–10 in order
│       ├── scripts/             # R pipeline (00–10)
│       ├── python/              # Python mirror pipeline (00–10)
│       ├── models/              # Trained XGBoost JSONs (< 10 MB each) + metrics
│       ├── figs/                # Generated figures (ignored)
│       └── output/              # Generated outputs (ignored)
├── scripts/
│   └── download_data.sh         # Fetches prediction dataset from Zenodo
├── requirements.txt             # Python dependencies
└── .gitignore
```

## Quick Start

```bash
git clone https://github.com/atmoschem/traffic_interpolation.git
cd traffic_interpolation
bash scripts/download_data.sh

# Point the pipeline at the directory holding the input data (see "Data" below)
export TRAFFIC_BASE=/path/to/utfpr2025

# Full R pipeline (steps 00–10)
Rscript experiments/2026_paper_xgboost_R/run_pipeline.R

# …or individual steps, e.g.
Rscript experiments/2026_paper_xgboost_R/scripts/00_train_sp_only.R
python experiments/2026_paper_xgboost_R/python/00_train_sp_only.py
```

If `TRAFFIC_BASE` is not set, scripts fall back to the hard-coded analysis
path and then to auto-detection relative to the script location.

## Data

The full SE Brazil prediction dataset is on **Zenodo** (DOI: [10.5281/zenodo.21342807](https://doi.org/10.5281/zenodo.21342807)):

- `SE_split.gpkg` — 200k OSM road segments
- `SE_full_predictions.csv.gz` — hourly PC, LCV, MC, Truck, Bus flows + speeds

The training/validation scripts expect the following inputs under `TRAFFIC_BASE`:

1. `xgboost_june_2026/training_data_fixed_rd.csv` — SP (ARTESP) training data
2. `data/cetrio/volumes e velocidades 2025-2024-2023-001.csv` — raw CET-Rio count data
3. `data/phase1/osm_se/SE_predicted.gpkg` — OSM+ split link segments with features
4. `data/flow_model/cetsptrans_with_tomtom.gpkg` — TomTom probe speed layers

Models are written to `$TRAFFIC_BASE/xgboost_june_2026/models/` and figures to
`$TRAFFIC_BASE/xgboost_june_2026/final/{rfigs,pyfigs}/`. Pre-trained models and
metrics JSONs are also committed under `experiments/2026_paper_xgboost_R/models/`.

## R pipeline (steps 00–10)

Run in order (models feed later steps; figures to `rfigs/`):

| Step | Script | Outputs |
|------|--------|---------|
| 00 | `00_train_sp_only.R` | SP-only flow models (Table 1 source), `results_sp_flow_r.json`, learning curves/validation/importance/RJ-total/DOW figs |
| 01 | `01_prepare_cetrio.R` | `cetrio_training_data_r.csv` (100,657 rows; 967 stations snapped to OSM+, ARTESP class profiles) |
| 02 | `02_retrain_and_figures.R` | SP+RJ flow models, `results_sprj_r.json`, RJ held-out validation, fig01–04 + fig06 |
| 03 | `03_build_speed_hourly.R` | Speed models (temporal R²≈0.92), `speed_hourly_results_r.json`, validation figs (≈ Fig 4) |
| 04 | `04_generate_all_shap_figures.R` | SHAP figs for 5 flow classes + speed (Fig 3, 5, S6–S8) |
| 05 | `05_plot_speed_validation.R` | Speed hexbin obs-vs-pred on ARTESP hourly rows |
| 06 | `06_regenerate_fig1_pvalues.R` | SP-only validation scatter with p-values (Fig 1) |
| 07 | `07_flow_error_distributions.R` | Flow residual histograms + Shapiro-Wilk (Fig 2) |
| 08 | `08_speed_residual_histogram.R` | Speed residual histogram, held-out 20% (Fig 6) |
| 09 | `09_gridded_maps.R` | Gridded prediction maps, 6 panels (Fig 8) |
| 10 | `10_supplementary_figures.R` | Station maps / weekly profiles / TomTom coverage (S1–S5) |

R packages: `xgboost`, `data.table`, `ggplot2`, `patchwork`, `terra`, `sf`,
`FNN`, `shapviz`, `hexbin`, `jsonlite`, `ggpmisc`.

## Python pipeline (mirror, steps 00–10)

`pip install -r requirements.txt`, then run `python/00_train_sp_only.py` …
`python/10_gridded_maps.py` in the same order. The Python pipeline has one
extra step (`03_validate_rj_proper.py`, strict cross-city validation) and
writes figures to `pyfigs/`.

## Manuscript figure ↔ script map

| Manuscript | Content | Python | R |
|---|---|---|---|
| Fig 1 | SP-only validation scatter (p-val) | 06 | 06 |
| Table 1 | SP-only flow metrics | 00 (`results_sp_flow.json`) | 00 (`results_sp_flow_r.json`) |
| Fig 2 | Flow error distributions + Shapiro | 08 | 07 |
| Fig 3 | SHAP PC | 07 | 04 |
| Fig 4 | Speed scatter by source | 04 | 03 |
| Fig 5 | SHAP speed | 07 | 04/05 |
| Fig 6 | Speed residual histogram + Shapiro | 09 | 08 |
| Fig 7 | SP+RJ validation scatter (p-val) | 05 (`fig02`) | 02 (`fig02`) |
| Fig 8 | Gridded maps (+speed) | 10 | 09 |
| S1–S5 | stations/profiles/coverage | — | 10 |
| S6–S8 | SHAP LCV/MC/Truck | 07 | 04 |

## R ↔ Python parity notes

- Same inputs everywhere: identical 23 features, fills, winsorization (p99), Mon–Wed/Thu–Sun splits, xgboost hyperparameters and seed 42.
- R-saved models carry `best_iteration`; R `predict()` honors it automatically, Python `Booster.predict` does not — the Python scripts pass `iteration_range=(0, best_iteration+1)` when loading R models.
- xgboost R and Python use different subsampling RNGs, so identical seeds give slightly different trees; SP+RJ metrics agree to < 0.01 R², SP-only to ≲ 0.02 R².

## Papers

| Year | Title | Directory |
|------|-------|-----------|
| 2026 | Predicting hourly vehicle type and speed at street level in South-East Brazil | `experiments/2026_paper_xgboost_R/` |
