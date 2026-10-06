#!/usr/bin/env python3
"""
Generate the final full-network prediction CSV with all flows and speeds.

Takes the SE_predicted.gpkg (which has all 22 static features already extracted),
loads the Mon-Wed paper models from xgboost_june_2026/models_monwed/,
predicts PC, LCV, MC, Truck for 24 hours × weekday/weekend,
computes BPR speed from capacity + predicted total flow,
adds the hourly XGBoost speed model (xgb_speed_hourly_r.json, R²≈0.92) for
weekday and weekend,
and outputs one clean CSV + the split OSM file (already exists).

Speed columns in the output:
  speed_bpr_{weekday,weekend}_h*       BPR curves from capacity and predicted flow
  speed_xgb_{weekday,weekend}_h*       hourly XGBoost speed model (this is the paper's model)
  speed_xgb_tomtom_static              legacy static segment value from the old TomTom-only model
  speed_avg                            legacy average of the two BPR means and the static value

Outputs:
  - SE_split.gpkg           (already exists — just referencing it)
  - SE_full_predictions.csv  (the master CSV with segment_id, features, all predictions)

Usage:
  micromamba run -n traffic_model python3 generate_full_se_predictions.py
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import geopandas as gpd
import time, os, warnings, json
warnings.filterwarnings('ignore')

BASE = "/home/sibarra/.hermes/profiles/aire/home/hermes_vein/utfpr2025"
SE_FEATURES = f"{BASE}/data/phase1/osm_se/SE_predicted.gpkg"
SPEED_GPKG   = f"{BASE}/data/phase1/osm_se/speed_xgb.gpkg"
MODEL_DIR    = f"{BASE}/xgboost_june_2026/models_monwed"
SPEED_MODEL_DIR = f"{BASE}/xgboost_june_2026/models"
SE_SPLIT     = f"{BASE}/data/phase1/osm_se/SE_split.gpkg"
OUT_CSV      = f"{BASE}/xgboost_june_2026/SE_full_predictions.csv"

# ── Feature definitions (matching paper training) ──
STATIC = ["lanes", "maxspeed", "nightlight", "pop_density",
          "road_density_500m", "road_density_1000m", "road_density_5000m",
          "highway_code", "surface_code", "oneway_code",
          "lu_tree", "lu_shrubland", "lu_grassland", "lu_cropland",
          "lu_builtup", "lu_bare", "lu_water", "lu_wetland",
          "lu_mangroves", "lu_moss"]
TEMPORAL = ["hour", "day_of_week", "is_weekend"]
FEATURES = STATIC + TEMPORAL

CLASSES = {
    "flow_pc":    {"short": "pc",    "label": "PC"},
    "flow_lcv":   {"short": "lcv",   "label": "LCV"},
    "flow_mc":    {"short": "mc",    "label": "MC"},
    "flow_truck": {"short": "truck", "label": "Truck"},
}

# Brazilian HCM-based capacity (veh/h per lane, by highway type)
CAP_PER_HIGHWAY = {
    "motorway": 2200, "trunk": 1800, "primary": 1400,
    "secondary": 900, "tertiary": 600, "unclassified": 600,
}
ALPHA, BETA = 0.15, 4.0  # BPR parameters

# Bus placeholder (per-road-type avg from SP EMME/2)
BUS_BY_HIGHWAY = {"primary": 15, "secondary": 12, "trunk": 8,
              "motorway": 5, "unclassified": 2}

HOURS = list(range(24))

# ── Weekend scaling factors from observed ARTESP data ──
# Mon-Wed models were trained on weekdays only, so they can't distinguish
# weekend patterns. We apply hour-by-hour scaling factors derived from
# the ARTESP observed data (Mon-Sun) to the weekday predictions.
# These capture: lower morning peak, flatter daytime, lower commercial traffic.
WEEKEND_FACTORS = {
"pc": {
    "hourly": [1.871, 1.438, 1.118, 0.930, 0.881, 0.885, 0.796, 0.527, 0.766,
               1.037, 1.094, 1.099, 1.140, 1.157, 1.161, 1.102, 0.876, 0.883,
               0.899, 0.966, 1.076, 1.212, 1.359, 1.517],
    "overall": 0.969,
},
"lcv": {
    "hourly": [0.901, 0.695, 0.462, 0.340, 0.260, 0.238, 0.226, 0.218, 0.352,
               0.533, 0.622, 0.647, 0.607, 0.623, 0.635, 0.561, 0.419, 0.329,
               0.295, 0.286, 0.339, 0.449, 0.598, 0.780],
    "overall": 0.521,
},
"mc": {
    "hourly": [0.827, 0.825, 0.804, 0.776, 0.761, 0.749, 0.689, 0.533, 0.763,
               0.912, 0.883, 0.898, 0.880, 0.854, 0.841, 0.786, 0.624, 0.550,
               0.527, 0.539, 0.609, 0.743, 0.842, 0.846],
    "overall": 0.806,
},
"truck": {
    "hourly": [0.643, 0.451, 0.316, 0.243, 0.199, 0.183, 0.171, 0.165, 0.264,
               0.428, 0.546, 0.594, 0.598, 0.598, 0.586, 0.539, 0.393, 0.299,
               0.260, 0.250, 0.286, 0.371, 0.494, 0.652],
    "overall": 0.522,
},
}

DAY_TYPES = [
{"label": "weekday", "dow": 2, "wknd": 0, "scale": None,            "speed_dows": [2],    "speed_wknd": 0},
{"label": "weekend", "dow": 2, "wknd": 0, "scale": "weekend",       "speed_dows": [5, 6], "speed_wknd": 1},
]


def r2_score(y_true, y_pred):
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)


def main():
    t_start = time.time()
    print("=" * 68)
    print("  Generate Full SE Network Predictions (Paper Models)")
    print("=" * 68)

    # ── 1. Load features ──
    print(f"\n[1] Loading SE feature data...")
    t0 = time.time()
    osm = gpd.read_file(SE_FEATURES)
    # Verify layer — only the 'SE_predicted' layer has features
    print(f"  {len(osm)} segments, {len(osm.columns)} columns")
    print(f"  CRS: {osm.crs}")
    # Add segment_id from index
    osm["segment_id"] = osm.index.values
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # ── 2. Load XGBoost speed ──
    print(f"\n[2] Loading XGBoost speed predictions...")
    t0 = time.time()
    speed_gdf = gpd.read_file(SPEED_GPKG)
    if "index_right" in speed_gdf.columns:
        speed_gdf = speed_gdf.drop(columns=["index_right"])
    osm = osm.join(speed_gdf[["speed_xgb"]], how="left")
    # Fill missing speed XGBoost with default (80% of maxspeed)
    default_speed_col = np.where(
        osm["maxspeed"].fillna(50).values * 0.8,
        osm["maxspeed"].fillna(50).values * 0.8,
        40
    )
    osm["speed_xgb"] = osm["speed_xgb"].fillna(pd.Series(default_speed_col))
    print(f"  Speed XGB: {osm['speed_xgb'].min():.1f}–{osm['speed_xgb'].max():.1f} km/h (mean={osm['speed_xgb'].mean():.1f})")
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # ── 3. Encode categories ──
    print(f"\n[3] Ensuring category encoding...")
    # Already encoded in SE_predicted.gpkg, but verify
    for c in ["highway_code", "surface_code", "oneway_code"]:
        nan_count = osm[c].isna().sum()
        if nan_count > 0:
            print(f"  Warning: {c} has {nan_count} NaN — filling with defaults")
    osm["highway_code"] = osm["highway_code"].fillna(4).astype(np.float32)
    osm["surface_code"] = osm["surface_code"].fillna(1).astype(np.float32)
    osm["oneway_code"]  = osm["oneway_code"].fillna(1).astype(np.float32)

    # Fill any other NaN in static features
    for c in STATIC:
        if c in osm.columns and osm[c].isna().any():
            fill_val = {"lanes": 2, "maxspeed": 60, "nightlight": 0, "pop_density": 0,
                        "road_density_500m": 0, "road_density_1000m": 0, "road_density_5000m": 0}.get(c, 0)
            osm[c] = osm[c].fillna(fill_val)
            print(f"  Filled NaN in {c} with {fill_val}")

    # ── 4. Compute segment capacity ──
    print(f"\n[4] Computing segment capacity...")
    lanes = np.maximum(osm["lanes"].fillna(2).values, 1)  # minimum 1 lane
    highway = osm["highway"].fillna("unclassified").values
    capacity = np.array([
        l * CAP_PER_HIGHWAY.get(hw, 600) for l, hw in zip(lanes, highway)
    ], dtype=np.float32)
    osm["capacity"] = capacity
    print(f"  Capacity: mean={capacity.mean():.0f}, min={capacity.min():.0f}, max={capacity.max():.0f}")

    # ── 5. Load models ──
    print(f"\n[5] Loading Mon-Wed paper models...")
    models = {}
    for cls, info in CLASSES.items():
        path = f"{MODEL_DIR}/xgb_flow_{info['short']}_monwed.json"
        if os.path.exists(path):
            m = xgb.Booster()
            m.load_model(path)
            models[cls] = m
            print(f"  ✓ {info['label']}: {path} ({m.num_features()} features)")
        else:
            print(f"  ✗ {info['label']}: NOT FOUND at {path}")

    # ── 6. Build static matrix ──
    print(f"\n[6] Building feature matrices...")
    static_data = np.column_stack([osm[c].values.astype(np.float32) for c in STATIC])
    print(f"  Static features: {static_data.shape}")

    # hourly speed model (paper model, 23 features). The R-saved file is preferred because
    # it is the model behind the manuscript figures; Python can load it and honours best_iteration
    sp_path = f"{SPEED_MODEL_DIR}/xgb_speed_hourly_r.json"
    if not os.path.exists(sp_path):
        sp_path = f"{SPEED_MODEL_DIR}/xgb_speed_hourly.json"
    speed_model = xgb.Booster()
    speed_model.load_model(sp_path)
    sp_best = int(speed_model.attributes().get("best_iteration", "0") or 0)
    sp_range = (0, sp_best + 1)
    print(f"  Speed model: {sp_path} ({speed_model.num_features()} features, best_iteration={sp_best})")

    def predict_speed_daytype(dows, wknd):
        """Hourly XGBoost speed for one day type, averaged over the given days of week."""
        hourly = np.zeros((len(osm), 24), dtype=np.float32)
        for hour in HOURS:
            acc = np.zeros(len(osm), dtype=np.float32)
            for dow in dows:
                hour_feats = np.zeros((len(osm), 3), dtype=np.float32)
                hour_feats[:, 0] = hour
                hour_feats[:, 1] = dow
                hour_feats[:, 2] = wknd
                X = np.column_stack([static_data, hour_feats])
                acc += speed_model.predict(xgb.DMatrix(X, feature_names=FEATURES),
                                           iteration_range=sp_range)
            hourly[:, hour] = acc / len(dows)
        return hourly

    # ── 7. Predict flows and speeds ──
    print(f"\n[7] Predicting flows and speeds...")
    t_pred = time.time()

    # Results containers: for each day_type, store (N, 24) arrays
    flow_by_dt = {}  # day_label -> {cls: (N, 24)}
    speed_by_dt = {}  # day_label -> (N, 24) BPR
    speed_xgb_hourly = {}  # day_label -> (N, 24) XGBoost

    # First pass: use dow=2, wknd=0 (Tuesday) as the base prediction for both
    # day types. Weekend gets scaled via observed ARTESP ratios.
    print(f"\n  --- BASE (Tuesday, applied to both) ---")
    base_flow = {}  # cls -> (N, 24)
    base_total = np.zeros((len(osm), 24), dtype=np.float32)

    for cls, info in CLASSES.items():
        t1 = time.time()
        cls_short = info["short"]
        hourly = np.zeros((len(osm), 24), dtype=np.float32)
        for hour in HOURS:
            hour_feats = np.zeros((len(osm), 3), dtype=np.float32)
            hour_feats[:, 0] = hour
            hour_feats[:, 1] = 2  # Tuesday (in training domain)
            hour_feats[:, 2] = 0  # weekday
            X = np.column_stack([static_data, hour_feats])
            hourly[:, hour] = models[cls].predict(
                xgb.DMatrix(X, feature_names=FEATURES)
            )
        base_flow[cls] = hourly
        base_total += hourly
        print(f"    {info['label']}: mean={hourly.mean():.1f} veh/h, {time.time()-t1:.1f}s")

    for dt in DAY_TYPES:
        label = dt["label"]
        print(f"\n  --- {label.upper()} ---")
        flow_by_dt[label] = {}
        total_hourly = np.zeros((len(osm), 24), dtype=np.float32)

        for cls, info in CLASSES.items():
            cls_short = info["short"]
            hourly = base_flow[cls].copy()
            if dt["scale"] == "weekend":
                factors = np.array(WEEKEND_FACTORS[cls_short]["hourly"], dtype=np.float32)
                hourly *= factors.reshape(1, 24)
            flow_by_dt[label][cls] = hourly
            total_hourly += hourly
            print(f"    {info['label']}: mean={hourly.mean():.1f} veh/h")

        # Bus placeholder (same for both, weekend scaled)
        bus_vals = np.array([
            BUS_BY_HIGHWAY.get(hw, 2) for hw in highway
        ], dtype=np.float32).reshape(-1, 1)
        bus_hourly = np.tile(bus_vals, (1, 24))
        if dt["scale"] == "weekend":
            bus_hourly *= 0.85  # Moderate weekend reduction for buses
        total_hourly += bus_hourly
        flow_by_dt[label]["flow_bus"] = bus_hourly
        print(f"    Bus: mean={bus_hourly.mean():.1f} (placeholder)")

        # BPR speed
        t1 = time.time()
        speed_hourly = np.zeros((len(osm), 24), dtype=np.float32)
        maxspeed_vals = osm["maxspeed"].fillna(50).values.astype(np.float32)
        for hour in HOURS:
            v_over_c = np.clip(total_hourly[:, hour] / np.clip(capacity, 1, None), 0, 10)
            speed_hourly[:, hour] = np.clip(
                maxspeed_vals / (1 + ALPHA * (v_over_c ** BETA)),
                5, maxspeed_vals
            )
        speed_by_dt[label] = speed_hourly
        print(f"    BPR Speed: mean={speed_hourly.mean():.1f} km/h, {time.time()-t1:.1f}s")

        # XGBoost hourly speed (paper model)
        t1 = time.time()
        speed_xgb_hourly[label] = predict_speed_daytype(dt["speed_dows"], dt["speed_wknd"])
        print(f"    XGBoost Speed: mean={speed_xgb_hourly[label].mean():.1f} km/h, {time.time()-t1:.1f}s")

    print(f"\n  Total prediction time: {time.time()-t_pred:.1f}s")

    # ── 8. Build output DataFrame ──
    print(f"\n[8] Building output CSV...")
    t0 = time.time()

    # Start with static identifiers
    out = pd.DataFrame({"segment_id": osm["segment_id"].values})

    # Basic OSM attributes for reference
    out["osm_id"] = osm["osm_id"].fillna(-1).values
    out["highway"] = osm["highway"].values
    out["lanes"] = np.maximum(osm["lanes"].fillna(2).values, 1)
    out["maxspeed"] = osm["maxspeed"].fillna(60).values
    out["surface"] = osm["surface"].fillna("unclassified").values
    out["oneway"] = osm["oneway"].fillna("no").values
    out["seg_len_m"] = osm["seg_len_m"].values

    # Features
    for c in STATIC:
        out[c] = osm[c].values

    # Capacity
    out["capacity"] = capacity

    # Speed XGBoost (legacy static value per segment, from the old TomTom-only model)
    out["speed_xgb_tomtom_static"] = osm["speed_xgb"].values

    # Hourly predictions per day type
    for dt in DAY_TYPES:
        label = dt["label"]
        for cls, info in CLASSES.items():
            cls_short = info["short"]
            hourly = flow_by_dt[label][cls]
            daily_mean = hourly.mean(axis=1)
            out[f"flow_{cls_short}_{label}"] = daily_mean
            for h in HOURS:
                out[f"flow_{cls_short}_{label}_h{h:02d}"] = hourly[:, h]

        # Bus daily average
        bus_hourly = flow_by_dt[label]["flow_bus"]
        out[f"flow_bus_{label}"] = bus_hourly.mean(axis=1)
        for h in HOURS:
            out[f"flow_bus_{label}_h{h:02d}"] = bus_hourly[:, h]

        # Speed BPR
        speed_hourly = speed_by_dt[label]
        out[f"speed_bpr_{label}"] = speed_hourly.mean(axis=1)
        for h in HOURS:
            out[f"speed_bpr_{label}_h{h:02d}"] = speed_hourly[:, h]

        # Speed XGBoost (hourly model)
        sp_hourly = speed_xgb_hourly[label]
        out[f"speed_xgb_{label}"] = sp_hourly.mean(axis=1)
        for h in HOURS:
            out[f"speed_xgb_{label}_h{h:02d}"] = sp_hourly[:, h]

    # Combined speed (legacy: BPR weekday + BPR weekend + static TomTom model)
    out["speed_avg"] = (out["speed_bpr_weekday"] + out["speed_bpr_weekend"] +
                        out["speed_xgb_tomtom_static"]) / 3

    # Total daily flow (all classes combined, weekday + weekend average)
    for dt in DAY_TYPES:
        label = dt["label"]
        flow_cols = [f"flow_{info['short']}_{label}" for info in CLASSES.values()]
        flow_cols.append(f"flow_bus_{label}")
        out[f"total_flow_{label}"] = out[flow_cols].sum(axis=1)

    print(f"  DataFrame built: {out.shape[0]} rows × {out.shape[1]} columns in {time.time()-t0:.1f}s")

    # ── 9. Save ──
    print(f"\n[9] Saving CSV...")
    t0 = time.time()
    out.to_csv(OUT_CSV, index=False)
    csv_size = os.path.getsize(OUT_CSV) / 1e6
    print(f"  Saved: {OUT_CSV}")
    print(f"  Size: {csv_size:.0f} MB")
    print(f"  Save time: {time.time()-t0:.1f}s")

    # ── 10. Quick validation summary ──
    print(f"\n[10] Summary statistics:")
    for dt in DAY_TYPES:
        label = dt["label"]
        print(f"\n  {label.upper()}:")
        for cls, info in CLASSES.items():
            col = f"flow_{info['short']}_{label}"
            vals = out[col]
            print(f"    {info['label']}: mean={vals.mean():.1f}, median={vals.median():.1f}, "
                  f"Q25={vals.quantile(0.25):.1f}, Q75={vals.quantile(0.75):.1f}, max={vals.max():.1f}")
        sp_col = f"speed_bpr_{label}"
        sp_vals = out[sp_col]
        print(f"    Speed BPR: mean={sp_vals.mean():.1f}, median={sp_vals.median():.1f} km/h")
        sp_x = out[f"speed_xgb_{label}"]
        print(f"    Speed XGBoost: mean={sp_x.mean():.1f}, median={sp_x.median():.1f} km/h")

    print(f"\n  Speed XGBoost (hourly model), column check:")
    for label in ["weekday", "weekend"]:
        for h in [2, 8, 18, 23]:
            v = out[f"speed_xgb_{label}_h{h:02d}"]
            print(f"    {label} h{h:02d}: mean={v.mean():.1f} km/h")
    print(f"\n  Legacy static TomTom speed: mean={out['speed_xgb_tomtom_static'].mean():.1f} km/h")
    print(f"  Capacity: mean={out['capacity'].mean():.0f} veh/h")

    # ── 11. Reference files ──
    split_size = os.path.getsize(SE_SPLIT) / 1e6
    print(f"\n{'=' * 68}")
    print(f"  DONE! Total time: {time.time()-t_start:.1f}s")
    print(f"{'=' * 68}")
    print(f"\n  Output files:")
    print(f"  1. CSV predictions:  {OUT_CSV} ({csv_size:.0f} MB)")
    print(f"  2. Split OSM gpkg:   {SE_SPLIT} ({split_size:.0f} MB)")
    print(f"\n  To load the CSV in R:")
    print(f'    library(data.table)')
    print(f'    dt <- fread("{OUT_CSV}")')
    print(f"\n  To associate with OSM geometry:")
    print(f'    library(sf)')
    print(f'    osm <- st_read("{SE_SPLIT}")')
    print(f'    osm$segment_id <- seq_len(nrow(osm)) - 1')
    print(f'    combined <- merge(osm, dt, by = "segment_id")')
    print(f"\n  To recompute R² (use validation CSVs from predictions/):")
    print(f'    sp_val <- fread("{BASE}/xgboost_june_2026/predictions/sp_test_obs_vs_pred.csv")')
    print(f'    rj_val <- fread("{BASE}/xgboost_june_2026/predictions/cetrio_obs_vs_pred.csv")')


if __name__ == "__main__":
    main()
