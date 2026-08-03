#!/usr/bin/env python3
"""
Step 9: Speed residuals histogram — same style as flow error histograms (Figure 2).
Manuscript Figure 6: Speed Model Residuals — Hourly.
Shapiro-Wilk test, normal curve overlay, R²/MAE annotations.
Rebuilds the combined hourly speed dataset (ARTESP + CET-Rio + TomTom),
then predicts with the trained hourly speed model on the held-out 20% test
set (same seeded split as 04_build_speed_hourly.py), so the metrics match
the speed validation figures.
"""
import os, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import geopandas as gpd
import xgboost as xgb
from scipy.spatial import cKDTree
from scipy.stats import shapiro, skew
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'font.size':10,'axes.titlesize':11,'axes.labelsize':10,
    'savefig.dpi':300,'savefig.bbox':'tight','axes.linewidth':0.8})

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
MODEL_DIR = f"{BASE}/xgboost_june_2026/models"
FIGS_DIR = f"{BASE}/xgboost_june_2026/final/pyfigs"
os.makedirs(FIGS_DIR, exist_ok=True)

STATIC = ['lanes','maxspeed','nightlight','pop_density','road_density_500m',
          'road_density_1000m','road_density_5000m','highway_code',
          'surface_code','oneway_code','lu_tree','lu_shrubland','lu_grassland',
          'lu_cropland','lu_builtup','lu_bare','lu_water','lu_wetland',
          'lu_mangroves','lu_moss']
TEMPORAL = ['hour','day_of_week','is_weekend']
FEATURES = STATIC + TEMPORAL
FILL_VALS = {'lanes':2.0,'maxspeed':60.0,'nightlight':0.0,'pop_density':0.0,
             'road_density_500m':0.0,'road_density_1000m':0.0,'road_density_5000m':0.0,
             'highway_code':4.0,'surface_code':1.0,'oneway_code':1.0}
for c in STATIC:
    if c.startswith('lu_'): FILL_VALS[c] = 0.0

print("="*60)
print("Speed Residuals Histogram (same style as flow error fig)")
print("="*60)

# ── 1. OSM grid ──
print("\n[1] Loading OSM+ grid...")
osm = gpd.read_file(OSM_GPKG, layer='SE_predicted',
                    columns=[c for c in ['geometry']+STATIC if c in ['geometry']+STATIC])
osm = osm.to_crs("EPSG:3857")
for c in STATIC:
    if c not in osm.columns: osm[c] = np.nan
osm_coords = np.column_stack([osm.geometry.centroid.x, osm.geometry.centroid.y])
tree = cKDTree(osm_coords)
print(f"  {len(osm)} segments")

# ── 2. ARTESP ──
print("\n[2] Loading ARTESP hourly speed...")
artesp_raw = pd.read_csv(TRAINING_CSV)
artesp = artesp_raw[(artesp_raw['source']=='artesp') & (artesp_raw['speed'].notna())].copy()
artesp['source'] = 'ARTESP'
for c in STATIC + TEMPORAL:
    if c not in artesp.columns: artesp[c] = FILL_VALS.get(c, 0)
    else: artesp[c] = artesp[c].fillna(FILL_VALS.get(c, 0))
print(f"  {len(artesp):,} obs")

# ── 3. CET-Rio ──
print("\n[3] Loading CET-Rio hourly speed...")
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
        keep['spd'] = pd.to_numeric(keep['spd'].str.replace(',','.'), errors='coerce')
        keep['lat'] = pd.to_numeric(keep['lat'].str.replace(',','.'), errors='coerce')
        keep['lon'] = pd.to_numeric(keep['lon'].str.replace(',','.'), errors='coerce')
        chunks.append(keep)
rj = pd.concat(chunks, ignore_index=True)
rj = rj.dropna(subset=['spd','lat','lon','hour','day_of_week']).copy()

stations = rj[['loc','lat','lon']].drop_duplicates('loc').reset_index(drop=True)
st_gdf = gpd.GeoDataFrame(stations, geometry=gpd.points_from_xy(stations['lon'], stations['lat']), crs="EPSG:4326")
st_gdf = st_gdf.to_crs("EPSG:3857")
st_coords = np.column_stack([st_gdf.geometry.x, st_gdf.geometry.y])
dists, idxs = tree.query(st_coords, k=1)
valid = dists <= 200
for c in STATIC: st_gdf[c] = np.nan
for i in np.where(valid)[0]:
    for c in STATIC:
        if c in osm.columns: st_gdf.loc[st_gdf.index[i], c] = osm.iloc[idxs[i]][c]
st_gdf = st_gdf[valid].reset_index(drop=True)
feat_map = st_gdf[['loc']+STATIC].set_index('loc')
rj = rj[rj['loc'].isin(feat_map.index)].copy()
for c in STATIC:
    rj[c] = rj['loc'].map(feat_map[c]).fillna(FILL_VALS[c])
rj.rename(columns={'spd':'speed'}, inplace=True)
rj['source'] = 'CET-Rio'
print(f"  {len(rj):,} obs")

# ── 4. TomTom ──
print("\n[4] Loading TomTom static speed...")
tt = gpd.read_file(TOMATO_GPKG, columns=['avg_speed','geometry'])
tt = tt[tt['avg_speed'].notna()].copy().to_crs("EPSG:3857")
tt_coords = np.column_stack([tt.geometry.centroid.x, tt.geometry.centroid.y])
dists_tt, idxs_tt = tree.query(tt_coords, k=1)
valid_tt = dists_tt <= 200
tt_list = []
for i in np.where(valid_tt)[0]:
    entry = {'speed': tt.iloc[i]['avg_speed'], 'source': 'TomTom'}
    for c in STATIC: entry[c] = osm.iloc[idxs_tt[i]][c]
    for t in TEMPORAL: entry[t] = np.nan
    tt_list.append(entry)
tt_df = pd.DataFrame(tt_list)
print(f"  {len(tt_df):,} segments")

# ── 5. Combine ──
print("\n[5] Building combined dataset...")
artesp_use = artesp[['speed','source']+FEATURES].copy()
rj_use = rj[['speed','source']+FEATURES].copy()
tt_use = tt_df[['speed','source']+FEATURES].copy()
combined = pd.concat([artesp_use.reset_index(drop=True),
                      rj_use.reset_index(drop=True),
                      tt_use.reset_index(drop=True)], ignore_index=True)
for c in STATIC:
    combined[c] = combined[c].fillna(FILL_VALS[c])
print(f"  Total: {len(combined):,} rows")
for src in combined['source'].unique():
    sub = combined[combined['source']==src]
    print(f"    {src:8s}: {len(sub):>7,} rows, speed mean={sub['speed'].mean():.1f}")

# ── 6. Load model + predict on the held-out test set ──
# Reproduces the exact 20% stratified split from 04_build_speed_hourly.py
# (same combined row order, random_state=42) so these metrics match the
# speed validation figures.
print("\n[6] Loading speed model and predicting (held-out test set)...")
model = xgb.Booster()
model.load_model(f"{MODEL_DIR}/xgb_speed_hourly.json")
src_map = {'ARTESP':0, 'CET-Rio':1, 'TomTom':2}
src_labels = combined['source'].map(src_map).values
_, test_idx = train_test_split(np.arange(len(combined)), test_size=0.2,
                               random_state=42, stratify=src_labels)
test = combined.iloc[test_idx]
X = test[FEATURES].values.astype(np.float32)
y_obs = test['speed'].values.astype(np.float32)
# Honor best_iteration from early stopping when present (matches R predict() default).
if model.best_iteration is not None:
    y_pred = model.predict(xgb.DMatrix(X, feature_names=FEATURES),
                           iteration_range=(0, model.best_iteration + 1))
else:
    y_pred = model.predict(xgb.DMatrix(X, feature_names=FEATURES))

residuals = y_pred - y_obs
r2 = r2_score(y_obs, y_pred)
mae = mean_absolute_error(y_obs, y_pred)
rmse = np.sqrt(np.mean(residuals**2))

# ── 7. Shapiro-Wilk (max 5000 samples) ──
rng = np.random.RandomState(42)
n_shap = min(5000, len(residuals))
idx_shap = rng.choice(len(residuals), n_shap, replace=False)
sw_stat, sw_p = shapiro(residuals[idx_shap])

print(f"\n  R² = {r2:.3f}")
print(f"  MAE = {mae:.1f} km/h")
print(f"  RMSE = {rmse:.1f} km/h")
print(f"  Residual mean = {residuals.mean():.1f}, sd = {residuals.std():.1f}")
print(f"  Skewness = {skew(residuals):.3f}")
print(f"  Shapiro-Wilk: W={sw_stat:.4f}, p={sw_p:.2e}")

# ── 8. Per-source breakdown ──
print(f"\n  Per-source metrics:")
for src in test['source'].unique():
    mask = (test['source'] == src).values
    r2_s = r2_score(y_obs[mask], y_pred[mask])
    mae_s = mean_absolute_error(y_obs[mask], y_pred[mask])
    bias_s = y_pred[mask].mean() / y_obs[mask].mean()
    print(f"    {src:8s}: R²={r2_s:.3f} MAE={mae_s:.1f} bias={bias_s:.2f}x n={mask.sum():,}")

# ── 9. Generate histogram ──
print("\n[7] Generating histogram...")
fig, ax = plt.subplots(figsize=(8, 6))

# Histogram
n_bins = 80
ax.hist(residuals, bins=n_bins, density=True, alpha=0.7,
        color='#2166AC', edgecolor='white', linewidth=0.3)

# Normal curve overlay
mu, sigma = residuals.mean(), residuals.std()
x_grid = np.linspace(residuals.min(), residuals.max(), 500)
ax.plot(x_grid, 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-(x_grid - mu)**2 / (2*sigma**2)),
        color='black', linewidth=1.5, linestyle='--', label='Normal distribution')

# Annotation box
sw_label = (f"Shapiro-Wilk\nW = {sw_stat:.4f}\np = {sw_p:.2e}")
stats_label = (f"R² = {r2:.3f}\nMAE = {mae:.1f} km/h\nRMSE = {rmse:.1f} km/h\n"
               f"Mean = {residuals.mean():.1f}\nSD = {residuals.std():.1f}\n"
               f"Skew = {skew(residuals):.3f}\nn = {len(residuals):,}")

props = dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='gray', linewidth=0.5)

ax.text(0.97, 0.97, sw_label, transform=ax.transAxes,
        fontsize=9, fontweight='bold', ha='right', va='top',
        bbox=props)
ax.text(0.03, 0.97, stats_label, transform=ax.transAxes,
        fontsize=8.5, ha='left', va='top',
        bbox=props)

ax.set_xlabel('Speed residual (predicted − observed, km/h)')
ax.set_ylabel('Density')
ax.set_title(f'Speed Model Residuals — Hourly (R²={r2:.3f}, MAE={mae:.1f} km/h)',
             fontweight='bold', pad=16)
ax.text(0.5, 1.015, 'Held-out test set (20%), all sources', transform=ax.transAxes,
        ha='center', va='bottom', fontsize=8.5, color='0.4')
ax.set_xlim(-60, 60)
ax.grid(True, alpha=0.2)

plt.tight_layout()
out_path = f'{FIGS_DIR}/fig08_speed_residual_histogram.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {out_path} ({os.path.getsize(out_path)/1024:.0f} KB)")
print("Done.")
