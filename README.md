# traffic_interpolation

Regressing and interpolating vehicular activity data across road networks to produce high-resolution inputs for the VEIN emissions model.

**Status:** Code and data for the XGBoost paper submission (2026).

---

## Structure

```
traffic_interpolation/
├── data/                   # Ignored — populated by download_data.sh
├── src/R/                  # Reusable functions (train, validate, plot)
├── experiments/
│   └── 2026_paper_xgboost_R/
│       ├── config.yaml     # Local data paths
│       ├── run_pipeline.R  # Master script
│       ├── models/         # XGBoost JSONs (< 10 MB each)
│       ├── scripts/        # Paper-specific scripts
│       └── figs/           # Generated figures
├── scripts/
│   ├── download_data.sh    # Fetches from Zenodo
│   └── split_data.sh       # Chunks CSVs for GitHub
├── config.default.yaml
├── requirements.txt
└── .gitignore
```

## Quick Start

```bash
git clone https://github.com/atmoschem/traffic_interpolation.git
cd traffic_interpolation
bash scripts/download_data.sh
cp config.default.yaml experiments/2026_paper_xgboost_R/config.yaml
# Edit config.yaml with your data paths
Rscript experiments/2026_paper_xgboost_R/run_pipeline.R
```

## Data

The full SE Brazil prediction dataset is on **Zenodo** (DOI pending):
- `SE_split.gpkg` — 200k OSM road segments
- `SE_full_predictions.csv.gz` — hourly PC, LCV, MC, Truck, Bus flows + speeds

## Papers

| Year | Title | Directory |
|------|-------|-----------|
| 2026 | Predicting hourly vehicle type and speed at street level in South-East Brazil | `experiments/2026_paper_xgboost_R/` |
