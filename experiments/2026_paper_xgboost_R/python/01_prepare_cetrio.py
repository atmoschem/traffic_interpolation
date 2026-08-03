#!/usr/bin/env python3
"""
Step 1: Prepare CET-Rio training data.
Reads CET-Rio CSV, snaps to OSM+, splits by ARTESP proportions,
saves as training CSV ready for model retrain.
"""
import os, sys, warnings, time
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree

# ── Dynamic Base Path Resolution ──
BASE = os.environ.get("TRAFFIC_BASE", "/home/sibarra/.hermes/profiles/aire/home/hermes_vein/utfpr2025")
if not os.path.exists(BASE) or not os.path.exists(os.path.join(BASE, "data")):
    # Try resolving relative to script location: final/python/01_prepare_cetrio.py -> final/python/ -> final/ -> xgboost_june_2026/ -> utfpr2025/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    potential_base = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
    if os.path.exists(os.path.join(potential_base, "data")):
        BASE = potential_base
    else:
        # Fallback to current working directory
        BASE = os.getcwd()

print(f"Using BASE path: {BASE}")

CETRIO_CSV = f"{BASE}/data/cetrio/volumes e velocidades 2025-2024-2023-001.csv"
OSM_GPKG = f"{BASE}/data/phase1/osm_se/SE_predicted.gpkg"
OUT_DIR = f"{BASE}/xgboost_june_2026"
os.makedirs(OUT_DIR, exist_ok=True)

STATIC_FEATURES = ['lanes','maxspeed','nightlight','pop_density','road_density_500m',
                   'road_density_1000m','road_density_5000m','highway_code',
                   'surface_code','oneway_code','lu_tree','lu_shrubland',
                   'lu_grassland','lu_cropland','lu_builtup','lu_bare',
                   'lu_water','lu_wetland','lu_mangroves','lu_moss']
FEATURES = STATIC_FEATURES + ['hour','day_of_week','is_weekend']

ARTESP_PCT = {
    'flow_pc': [64.2,56.3,50.4,47.1,51.9,64.5,73.1,75.9,78.3,74.7,73.7,74.2,
                75.1,75.0,74.4,74.7,76.0,77.7,78.3,78.4,76.7,75.3,73.9,70.1],
    'flow_mc': [7.4,8.0,8.3,7.5,6.9,7.3,7.3,6.8,9.5,5.2,5.1,5.3,
                5.4,5.4,5.4,5.3,5.5,6.2,6.0,5.3,5.4,5.8,6.4,6.5],
    'flow_lcv': [19.9,25.0,28.9,31.8,28.9,19.7,13.8,12.1,10.5,14.1,14.8,14.4,
                 13.7,13.7,14.2,14.0,12.9,11.3,11.0,11.4,12.6,13.2,13.8,16.4],
    'flow_truck': [8.5,10.7,12.4,13.6,12.3,8.4,5.9,5.2,1.7,6.0,6.3,6.1,
                   5.8,5.8,6.0,6.0,5.5,4.8,4.7,4.9,5.4,5.7,5.9,7.0]
}
FILL_VALS = {'lanes':2.0,'maxspeed':60.0,'nightlight':0.0,'pop_density':0.0,
             'road_density_500m':0.0,'road_density_1000m':0.0,'road_density_5000m':0.0,
             'highway_code':4.0,'surface_code':1.0,'oneway_code':1.0}
for c in STATIC_FEATURES:
    if c.startswith('lu_'): FILL_VALS[c] = 0.0

TARGETS = ['flow_pc','flow_mc','flow_lcv','flow_truck']

print("="*60)
print("Prepare CET-Rio Training Data (Organized Pipeline)")
print("="*60)

# ── 1. Read CET-Rio ──
print("\n[1/6] Reading CET-Rio CSV...")
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
df = pd.concat(chunks, ignore_index=True)
df = df.dropna(subset=['vol','lat','lon'])
print(f"  {len(df):,} obs | {df['loc'].nunique()} stations | {time.time()-t0:.0f}s")

# ── 2. Station coords ──
print("\n[2/6] Extracting station coordinates...")
stations = df[['loc','lat','lon']].drop_duplicates('loc').reset_index(drop=True)
print(f"  {len(stations)} stations")

# ── 3. Load OSM+ ──
print("\n[3/6] Loading OSM+ features...")
osm = gpd.read_file(OSM_GPKG, layer='SE_predicted',
                    columns=[c for c in ['geometry']+STATIC_FEATURES if c in 
                            ['geometry']+STATIC_FEATURES])
osm = osm.to_crs("EPSG:3857")
if 'geometry' not in osm.columns:
    osm = gpd.read_file(OSM_GPKG, layer='SE_predicted')
osm = osm[['geometry']+[c for c in STATIC_FEATURES if c in osm.columns]].to_crs("EPSG:3857")
print(f"  {len(osm)} OSM segments")

# ── 4. Snap ──
print("\n[4/6] Snapping stations to OSM+...")
st_gdf = gpd.GeoDataFrame(stations, geometry=gpd.points_from_xy(stations['lon'], stations['lat']), crs="EPSG:4326")
st_gdf = st_gdf.to_crs("EPSG:3857")
osm_coords = np.column_stack([osm.geometry.centroid.x, osm.geometry.centroid.y])
osm_tree = cKDTree(osm_coords)
st_coords = np.column_stack([st_gdf.geometry.x, st_gdf.geometry.y])
dists, idxs = osm_tree.query(st_coords, k=1)
valid = dists <= 200
print(f"  Matched: {valid.sum()}/{len(st_gdf)}")
for c in STATIC_FEATURES:
    st_gdf[c] = np.nan
for i in np.where(valid)[0]:
    for c in STATIC_FEATURES:
        if c in osm.columns:
            v = osm.iloc[idxs[i]][c]
            st_gdf.loc[st_gdf.index[i], c] = v
st_gdf = st_gdf[valid].reset_index(drop=True)
print(f"  {len(st_gdf)} stations with features")

# ── 5. Merge features + split flow ──
print("\n[5/6] Building training rows...")
feat_map = st_gdf[['loc']+STATIC_FEATURES].set_index('loc')
obs_all = df[df['loc'].isin(feat_map.index)].copy()
# Add features
for c in STATIC_FEATURES:
    obs_all[c] = obs_all['loc'].map(feat_map[c])
    obs_all[c] = obs_all[c].fillna(FILL_VALS[c])

rows = []
for _, row in obs_all.iterrows():
    h = int(row['hour']) % 24
    total = row['vol']
    classes = {t: total * ARTESP_PCT[t][h] / 100 for t in TARGETS}
    entry = {f: row[f] for f in STATIC_FEATURES}
    entry['hour'] = float(row['hour'])
    entry['day_of_week'] = float(row['day_of_week'])
    entry['is_weekend'] = float(row['is_weekend'])
    entry.update(classes)
    rows.append(entry)

rj_train = pd.DataFrame(rows)
print(f"  {len(rj_train):,} training rows from CET-Rio")

if len(rj_train) == 0:
    sys.exit("ERROR: 0 training rows built (station snapping matched nothing — "
             "check PROJ/GDAL installation and input data). Refusing to write empty CSV.")

# ── 6. Save ──
print("\n[6/6] Saving...")
rj_train.to_csv(f'{OUT_DIR}/cetrio_training_data.csv', index=False)
print(f"  Saved: {OUT_DIR}/cetrio_training_data.csv")
print(f"  Columns: {list(rj_train.columns)}")
for t in TARGETS:
    print(f"  {t}: mean={rj_train[t].mean():.1f} max={rj_train[t].max():.0f}")
print(f"\nDone in {time.time()-t0:.0f}s")
