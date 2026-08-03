#!/usr/bin/env python3
"""
Step 4: Build speed model using hourly data from ARTESP + CET-Rio + static TomTom.
Adds temporal features (hour, day_of_week) and volume as optional feature.
Compares against the old static-only model.
"""
import os, warnings, time, json
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import geopandas as gpd
import xgboost as xgb
from scipy.spatial import cKDTree
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({'font.size':9,'axes.titlesize':10,'axes.labelsize':9.5,
    'savefig.dpi':300,'savefig.bbox':'tight','axes.linewidth':0.8,
    'axes.grid':False})

# ── Dynamic Base Path Resolution ──
BASE = os.environ.get("TRAFFIC_BASE", "/home/sibarra/.hermes/profiles/aire/home/hermes_vein/utfpr2025")
if not os.path.exists(BASE) or not os.path.exists(os.path.join(BASE, "data")):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    potential_base = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
    if os.path.exists(os.path.join(potential_base, "data")):
        BASE = potential_base
    else:
        BASE = os.getcwd()

print(f"Using BASE path: {BASE}")

CETRIO_CSV = f"{BASE}/data/cetrio/volumes e velocidades 2025-2024-2023-001.csv"
TRAINING_CSV = f"{BASE}/xgboost_june_2026/training_data_fixed_rd.csv"
OSM_GPKG = f"{BASE}/data/phase1/osm_se/SE_predicted.gpkg"
TOMATO_GPKG = f"{BASE}/data/flow_model/cetsptrans_with_tomtom.gpkg"
OUT_DIR = f"{BASE}/xgboost_june_2026"
MODEL_DIR = f"{OUT_DIR}/models"
FIGS_DIR = f"{OUT_DIR}/final/pyfigs"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)

STATIC = ['lanes','maxspeed','nightlight','pop_density','road_density_500m',
          'road_density_1000m','road_density_5000m','highway_code',
          'surface_code','oneway_code','lu_tree','lu_shrubland','lu_grassland',
          'lu_cropland','lu_builtup','lu_bare','lu_water','lu_wetland',
          'lu_mangroves','lu_moss']
TEMPORAL = ['hour','day_of_week','is_weekend']
FEATURES_STATIC = STATIC
FEATURES_TEMPORAL = STATIC + TEMPORAL
FILL_VALS = {'lanes':2.0,'maxspeed':60.0,'nightlight':0.0,'pop_density':0.0,
             'road_density_500m':0.0,'road_density_1000m':0.0,'road_density_5000m':0.0,
             'highway_code':4.0,'surface_code':1.0,'oneway_code':1.0}
for c in STATIC:
    if c.startswith('lu_'): FILL_VALS[c] = 0.0

PARAMS = {'objective':'reg:squarederror','max_depth':8,'learning_rate':0.08,
          'subsample':0.8,'colsample_bytree':0.8,'min_child_weight':5,
          'gamma':1,'seed':42,'n_jobs':-1}

def save_fig(name):
    path = f'{FIGS_DIR}/{name}'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

print("="*60)
print("Hourly Speed Model: ARTESP + CET-Rio hourly, TomTom static (Organized)")
print("="*60)

# ── 1. Load OSM+ grid ──
print("\n[1] Loading OSM+ feature grid...")
osm = gpd.read_file(OSM_GPKG, layer='SE_predicted',
                    columns=[c for c in ['geometry']+STATIC if c in ['geometry']+STATIC])
osm = osm.to_crs("EPSG:3857")
for c in STATIC:
    if c not in osm.columns:
        osm[c] = np.nan
print(f"  {len(osm)} segments")

# ── 2. ARTESP hourly ──
print("\n[2] Loading ARTESP hourly speed...")
t0 = time.time()
artesp_raw = pd.read_csv(TRAINING_CSV)
artesp = artesp_raw[(artesp_raw['source']=='artesp') & (artesp_raw['speed'].notna())].copy()
artesp['source'] = 'ARTESP'
n_artesp = len(artesp)
print(f"  {n_artesp:,} hourly obs | {artesp['snap_dist_m'].nunique()} stations | {time.time()-t0:.0f}s")

# ── 3. CET-Rio hourly ──
print(f"\n[3] Loading CET-Rio hourly speed...")
t0 = time.time()
chunks = []
reader = pd.read_csv(CETRIO_CSV, sep=';', encoding='utf-8-sig', header=None,
                     chunksize=500000, names=['loc','ts','vol','spd','lat','lon'])
for chunk in reader:
    chunk['ts_str'] = chunk['ts'].astype(str)
    mask = chunk['ts_str'].str[:8].between('20240304','20240310')
    keep = chunk[mask].copy()
    if len(keep) > 0:
        keep['hour'] = keep['ts_str'].str[8:10].astype(int)
        keep['date_str'] = keep['ts_str'].str[:8]
        keep['date'] = pd.to_datetime(keep['date_str'], format='%Y%m%d')
        keep['day_of_week'] = keep['date'].dt.dayofweek
        keep['is_weekend'] = (keep['day_of_week'] >= 5).astype(int)
        keep['vol'] = pd.to_numeric(keep['vol'], errors='coerce')
        keep['spd'] = pd.to_numeric(keep['spd'].str.replace(',','.'), errors='coerce')
        keep['lat'] = pd.to_numeric(keep['lat'].str.replace(',','.'), errors='coerce')
        keep['lon'] = pd.to_numeric(keep['lon'].str.replace(',','.'), errors='coerce')
        chunks.append(keep)
rj_raw = pd.concat(chunks, ignore_index=True)
rj = rj_raw.dropna(subset=['spd','lat','lon', 'hour', 'day_of_week']).copy()
print(f"  {len(rj):,} hourly obs | {rj['loc'].nunique()} stations | {time.time()-t0:.0f}s")

# ── 4. Snap CET-Rio to OSM+ ──
print("\n[4] Snapping CET-Rio to OSM+...")
t0 = time.time()
stations = rj[['loc','lat','lon']].drop_duplicates('loc').reset_index(drop=True)
st_gdf = gpd.GeoDataFrame(stations, geometry=gpd.points_from_xy(stations['lon'], stations['lat']), crs="EPSG:4326")
st_gdf = st_gdf.to_crs("EPSG:3857")
osm_coords = np.column_stack([osm.geometry.centroid.x, osm.geometry.centroid.y])
tree = cKDTree(osm_coords)
st_coords = np.column_stack([st_gdf.geometry.x, st_gdf.geometry.y])
dists, idxs = tree.query(st_coords, k=1)
valid = dists <= 200
print(f"  Snapped: {valid.sum()}/{len(st_gdf)} stations (within 200m)")
for c in STATIC:
    st_gdf[c] = np.nan
for i in np.where(valid)[0]:
    for c in STATIC:
        if c in osm.columns:
            st_gdf.loc[st_gdf.index[i], c] = osm.iloc[idxs[i]][c]
st_gdf = st_gdf[valid].reset_index(drop=True)
feat_map = st_gdf[['loc']+STATIC].set_index('loc')
rj = rj[rj['loc'].isin(feat_map.index)].copy()
for c in STATIC:
    rj[c] = rj['loc'].map(feat_map[c]).fillna(FILL_VALS[c])
rj.rename(columns={'spd':'speed'}, inplace=True)
rj['source'] = 'CET-Rio'
print(f"  {len(rj):,} hourly obs after snapping | {time.time()-t0:.0f}s")

# ── 5. TomTom ──
print("\n[5] Loading TomTom static speed...")
t0 = time.time()
tt = gpd.read_file(TOMATO_GPKG, columns=['avg_speed','geometry'])
tt = tt[tt['avg_speed'].notna()].copy()
tt = tt.to_crs("EPSG:3857")
tt_coords = np.column_stack([tt.geometry.centroid.x, tt.geometry.centroid.y])
dists_tt, idxs_tt = tree.query(tt_coords, k=1)
valid_tt = dists_tt <= 200
tt_list = []
for i in np.where(valid_tt)[0]:
    entry = {'speed': tt.iloc[i]['avg_speed'], 'source': 'TomTom'}
    for c in STATIC:
        entry[c] = osm.iloc[idxs_tt[i]][c]
    for t in TEMPORAL:
        entry[t] = np.nan
    tt_list.append(entry)
tt_df = pd.DataFrame(tt_list)
print(f"  {len(tt_df):,} segments | {time.time()-t0:.0f}s")

# ── 6. Combine datasets ──
print("\n[6] Building combined dataset...")
ALL_COLS = ['speed','source'] + FEATURES_TEMPORAL

artesp_use = artesp[ALL_COLS].copy()
rj_use = rj[ALL_COLS].copy()
tt_use = tt_df[ALL_COLS].copy()

for c in FEATURES_TEMPORAL:
    artesp_use[c] = pd.to_numeric(artesp_use[c], errors='coerce')
    rj_use[c] = pd.to_numeric(rj_use[c], errors='coerce')
    tt_use[c] = pd.to_numeric(tt_use[c], errors='coerce')
tt_use['speed'] = pd.to_numeric(tt_use['speed'], errors='coerce')

combined = pd.concat([artesp_use, rj_use, tt_use], ignore_index=True, copy=False)
for c in STATIC:
    combined[c] = combined[c].fillna(FILL_VALS[c])

print(f"\n  Combined dataset: {len(combined):,} rows")
for src in combined['source'].unique():
    sub = combined[combined['source']==src]
    print(f"    {src:8s}: {len(sub):>7,} rows, speed mean={sub['speed'].mean():.1f}")

# ── 7. Train: STATIC ONLY (baseline) vs TEMPORAL ──
print("\n[7] Training XGBoost models...")
models = {}
results = {}

for variant, features in [('static_only', FEATURES_STATIC), ('temporal', FEATURES_TEMPORAL)]:
    print(f"\n  --- {variant} ({len(features)} features) ---")
    t0 = time.time()
    
    X = combined[features].values.astype(np.float32)
    y = combined['speed'].values.astype(np.float32)
    
    src_map = {'ARTESP':0, 'CET-Rio':1, 'TomTom':2}
    src_labels = combined['source'].map(src_map).values
    
    train_idx, test_idx = train_test_split(
        np.arange(len(combined)), test_size=0.2, random_state=42,
        stratify=src_labels
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    test_sources = combined['source'].values[test_idx]
    
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=features)
    
    evals_result = {}
    model = xgb.train(PARAMS, dtrain, num_boost_round=500,
                      evals=[(dtrain,'train'),(dtest,'test')],
                      early_stopping_rounds=30, evals_result=evals_result,
                      verbose_eval=False)
    
    best_round = model.best_iteration
    y_pred = model.predict(dtest)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    models[variant] = model
    results[variant] = {
        'r2': round(r2,4), 'mae': round(mae,1), 'rmse': round(rmse,1),
        'rounds': best_round, 'n_train': len(y_train), 'n_test': len(y_test),
        'features': features,
        'evals': {'train_rmse': evals_result['train']['rmse'],
                  'test_rmse': evals_result['test']['rmse']}
    }
    
    print(f"    R² = {r2:.4f}")
    print(f"    MAE = {mae:.1f} km/h")
    print(f"    RMSE = {rmse:.1f} km/h")
    
    print(f"\n    Per-source test metrics:")
    for src in combined['source'].unique():
        src_mask = test_sources == src
        if src_mask.sum() > 10:
            y_src = y_test[src_mask]
            X_src = X_test[src_mask]
            p_src = model.predict(xgb.DMatrix(X_src, feature_names=features))
            r2_s = r2_score(y_src, p_src)
            bias_s = p_src.mean() / y_src.mean() if y_src.mean() > 0 else np.nan
            print(f"      {src:8s}: R²={r2_s:.3f} bias={bias_s:.2f}x n={src_mask.sum():,} obs={y_src.mean():.1f}")
    print(f"    Time: {time.time()-t0:.0f}s")

# ── 8. Save hourly model ──
print("\n[8] Saving models...")
model_path = f'{MODEL_DIR}/xgb_speed_hourly.json'
models['temporal'].save_model(model_path)
print(f"  Hourly model: {model_path}")

model_path_old = f'{MODEL_DIR}/xgb_speed_hourly_static.json'
models['static_only'].save_model(model_path_old)
print(f"  Static-only (baseline): {model_path_old}")

with open(f'{MODEL_DIR}/speed_hourly_results.json','w') as f:
    json.dump(results, f, indent=2)

# ── 9. Figures ──
print("\n[9] Generating comparison figures...")

# Fig 1: Learning curves side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for idx, (variant, label, color) in enumerate([
    ('static_only', 'Static-only (20 features)', '#636363'),
    ('temporal', 'Temporal (23 features)', '#2166AC')
]):
    evals = results[variant]['evals']
    ax = axes[idx]
    ax.plot(range(1, len(evals['train_rmse'])+1), evals['train_rmse'],
            color='#999999', alpha=0.4, lw=0.7, label='Train')
    ax.plot(range(1, len(evals['test_rmse'])+1), evals['test_rmse'],
            color=color, lw=1.2, label='Test')
    ax.axvline(results[variant]['rounds'], color='red', ls='--', alpha=0.4, lw=0.7)
    ax.set_xlabel('Boosting round')
    ax.set_ylabel('RMSE (km/h)')
    r2_val = results[variant]['r2']
    ax.set_title(f'{label}\nR²={r2_val:.3f}', fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)
fig.suptitle('Speed Model: Static vs Temporal Features', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
save_fig('fig_speed_hourly_comparison.png')

# Fig 2: Validation scatter — temporal model with source colors, held-out
# test set (same seeded split as training, so the cloud matches the title R²)
fig, ax = plt.subplots(figsize=(6.5, 5.5))
src_colors = {'ARTESP':'#2166AC','CET-Rio':'#D6604D','TomTom':'#4DAF4A'}
model = models['temporal']
features = FEATURES_TEMPORAL
_, test_idx = train_test_split(
    np.arange(len(combined)), test_size=0.2, random_state=42,
    stratify=combined['source'].map({'ARTESP':0,'CET-Rio':1,'TomTom':2}).values)
test = combined.iloc[test_idx]
X_test = test[features].values.astype(np.float32)
y_test = test['speed'].values
y_pred_test = model.predict(xgb.DMatrix(X_test, feature_names=features))
for src in test['source'].unique():
    mask = (test['source'] == src).values
    ax.scatter(y_test[mask], y_pred_test[mask], c=src_colors.get(src,'gray'),
               s=4, alpha=0.2, label=f'{src} ({mask.sum():,})', edgecolors='none')
lims = [0, max(y_test.max(), y_pred_test.max()) * 1.02]
ax.plot(lims, lims, '--', color='#333', alpha=0.3, lw=0.6)
ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect('equal')
ax.set_xlabel('Observed speed (km/h)'); ax.set_ylabel('Predicted speed (km/h)')
r = results['temporal']['r2']
ax.set_title(f'Hourly Speed Model: R²={r:.3f}', fontweight='bold')
leg = ax.legend(fontsize=7, markerscale=3)
for h in leg.legend_handles:
    h.set_alpha(1.0)
ax.grid(True, alpha=0.2)
plt.tight_layout()
save_fig('fig_speed_hourly_validation.png')

# Fig 3: Feature importance comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
for idx, (variant, color) in enumerate([('static_only','#636363'),('temporal','#2166AC')]):
    m = models[variant]
    imp = m.get_score(importance_type='gain')
    items = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:15]
    names = [x[0] for x in items[::-1]]
    vals = [x[1] for x in items[::-1]]
    ax = axes[idx]
    ax.barh(range(len(names)), vals, color=color, alpha=0.8, height=0.65)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7.5)
    ax.set_xlabel('Gain')
    ax.set_title(f'{variant} ({len(names)} features)', fontweight='bold')
    ax.grid(True, alpha=0.2, axis='x')
fig.suptitle('Feature Importance Comparison', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
save_fig('fig_speed_hourly_importance.png')

print(f"\nDone. Hourly speed model saved: {MODEL_DIR}/xgb_speed_hourly.json")
