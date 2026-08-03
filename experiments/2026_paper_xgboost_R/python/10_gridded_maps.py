#!/usr/bin/env python3
"""
Step 10 (Python): 2D histogram gridded maps for all flow classes + speed.
Manuscript Figure 8 — "Predicted Traffic Flow & Speed — Weekday 8am".
Style reference: Truck map sent by user — inferno (low=yellow, high=dark),
state boundaries, city labels (SP,RJ,BH,VIT,CTBA), clean axes.

Port of scripts/plot_gridded_maps.py. Flow models: xgb_flow_*_sp.json with
fallback to R-saved xgb_flow_*_sp_r.json (xgboost loads R-saved JSON fine).
Speed model: xgb_speed_hourly.json with fallback to xgb_speed_hourly_r.json;
the Speed panel is optional — it is skipped with a printed warning if the
model file is missing or unreadable (e.g. while being retrained).
"""
import os, warnings, json
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import geopandas as gpd
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import Normalize
from shapely.geometry import box
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import natural_earth

rcParams.update({'font.size':14,'axes.titlesize':15,'axes.labelsize':13,
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

print("Using BASE path:", BASE)

OSM_GPKG = f"{BASE}/data/phase1/osm_se/SE_predicted.gpkg"
MODEL_DIR = f"{BASE}/xgboost_june_2026/models"
FIGS_DIR = f"{BASE}/xgboost_june_2026/final/pyfigs"
os.makedirs(FIGS_DIR, exist_ok=True)

STATIC = ['lanes','maxspeed','nightlight','pop_density',
          'road_density_500m','road_density_1000m','road_density_5000m',
          'highway_code','surface_code','oneway_code',
          'lu_tree','lu_shrubland','lu_grassland','lu_cropland',
          'lu_builtup','lu_bare','lu_water','lu_wetland','lu_mangroves','lu_moss']
TEMPORAL = ['hour','day_of_week','is_weekend']
FEATURES = STATIC + TEMPORAL
FILL_VALS = {'lanes':2,'maxspeed':60,'nightlight':0,'pop_density':0,
             'road_density_500m':0,'road_density_1000m':0,'road_density_5000m':0,
             'highway_code':4,'surface_code':1,'oneway_code':1}
for c in STATIC:
    if c.startswith('lu_'): FILL_VALS[c] = 0

# 'files' is ordered: primary choice first, fallbacks after.
CLASSES = {
    'PC':    {'files':['xgb_flow_pc_sp.json',    'xgb_flow_pc_sp_r.json'],    'label':'PC','unit':'veh/h','vmax':5000, 'cmap':'inferno_r'},
    'LCV':   {'files':['xgb_flow_lcv_sp.json',   'xgb_flow_lcv_sp_r.json'],   'label':'LCV','unit':'veh/h','vmax':400, 'cmap':'inferno_r'},
    'MC':    {'files':['xgb_flow_mc_sp.json',    'xgb_flow_mc_sp_r.json'],    'label':'MC','unit':'veh/h','vmax':400, 'cmap':'inferno_r'},
    'Truck': {'files':['xgb_flow_truck_sp.json', 'xgb_flow_truck_sp_r.json'], 'label':'Trucks','unit':'veh/h','vmax':300, 'cmap':'inferno_r'},
    'Bus':   {'files':['xgb_flow_bus_sp.json',   'xgb_flow_bus_sp_r.json'],   'label':'Buses','unit':'veh/h','vmax':60, 'cmap':'inferno_r'},
    'Speed': {'files':['xgb_speed_hourly.json',  'xgb_speed_hourly_r.json'],  'label':'Speed','unit':'km/h','vmax':110, 'cmap':'viridis_r'},
}

# City labels (lat, lon)
CITIES = {
    'SP': (-23.55, -46.63),
    'RJ': (-22.91, -43.20),
    'BH': (-19.92, -43.94),
    'VIT': (-20.32, -40.34),
    'CTBA': (-25.43, -49.27),
}

print("="*60)
print("2D Histogram Gridded Maps — All Classes (Python)")
print("="*60)

# ── 1. Load OSM grid ──
print("\n[1] Loading OSM+ grid...")
osm = gpd.read_file(OSM_GPKG, layer='SE_predicted')
osm = osm.to_crs("EPSG:4326")
for c in STATIC:
    if c not in osm.columns: osm[c] = FILL_VALS.get(c, 0)
    osm[c] = osm[c].fillna(FILL_VALS.get(c, 0))
osm['hour'] = 8
osm['day_of_week'] = 1
osm['is_weekend'] = 0

# Centroids for gridding
cents = osm.geometry.centroid
lon = cents.x.values
lat = cents.y.values
print(f"  {len(osm)} segments")

# ── 2. State boundaries ──
print("\n[2] Loading state boundaries...")
try:
    states = gpd.read_file('/tmp/brazil_states.geojson')
except:
    import urllib.request
    url = "https://raw.githubusercontent.com/codeforgermany/click_that_hood/main/public/data/brazil-states.geojson"
    urllib.request.urlretrieve(url, '/tmp/brazil_states.geojson')
    states = gpd.read_file('/tmp/brazil_states.geojson')
se_states = ['São Paulo','Rio de Janeiro','Minas Gerais','Espírito Santo','Paraná','Santa Catarina']
se = states[states['name'].isin(se_states)].to_crs("EPSG:4326")
print(f"  {len(se)} states")

# ── 3. Predict ──
print("\n[3] Loading models and predicting...")
X = osm[FEATURES].values.astype(np.float32)
dm = xgb.DMatrix(X, feature_names=FEATURES)

predictions = {}
for abbr, cls in CLASSES.items():
    path = None
    for fname in cls['files']:
        cand = f"{MODEL_DIR}/{fname}"
        if os.path.exists(cand):
            path = cand
            break
    if path is None:
        print(f"  WARNING: no model file found for {abbr} ({cls['files']}), skipping panel")
        continue
    try:
        model = xgb.Booster()
        model.load_model(path)
        # R-saved models store best_iteration; slice trees to match R predict()
        best_iter = model.attr('best_iteration')
        iteration_range = (0, int(best_iter) + 1) if best_iter is not None else (0, 0)
        pred = model.predict(dm, iteration_range=iteration_range)
    except Exception as e:
        print(f"  WARNING: could not load/predict with {path} ({e}), skipping panel")
        continue
    pred = np.maximum(pred, 0)  # clip negatives
    if abbr == 'Speed':
        pred = np.clip(pred, 0, cls['vmax'])
    predictions[abbr] = pred
    print(f"  {cls['label']:8s}: mean={pred.mean():.0f} max={pred.max():.0f} {cls['unit']}  [{os.path.basename(path)}]")

# ── 4. Gridding function ──
print("\n[4] Creating 2D histogram grids...")
RES = 0.03
xmin, xmax = lon.min(), lon.max()
ymin, ymax = lat.min(), lat.max()
nx = int((xmax - xmin) / RES) + 1
ny = int((ymax - ymin) / RES) + 1
print(f"  Grid: {nx}x{ny} cells")

lon_bins = np.linspace(xmin, xmax, nx+1)
lat_bins = np.linspace(ymin, ymax, ny+1)
lon_idx = np.clip(np.searchsorted(lon_bins, lon, side='right') - 1, 0, nx-1)
lat_idx = np.clip(np.searchsorted(lat_bins, lat, side='right') - 1, 0, ny-1)
cell_idx = lat_idx * nx + lon_idx

def make_grid(vals):
    """Create 2D histogram grid with shape (ny, nx) = (lat, lon) for imshow."""
    grid = np.full((ny, nx), np.nan)
    sums = np.bincount(cell_idx, weights=vals, minlength=nx*ny)
    counts = np.bincount(cell_idx, minlength=nx*ny)
    mask = counts > 0
    grid.flat[mask] = sums[mask] / counts[mask]
    return grid

grids = {k: make_grid(v) for k, v in predictions.items()}

# ── 5. Plot function ──
print("\n[5] Generating maps...")

def plot_class(abbr, grid_data, cls, figsize=(8, 7)):
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': ccrs.PlateCarree()})
    
    vmax = cls['vmax']
    data = grid_data.copy()
    data[data > vmax] = vmax
    
    im = ax.imshow(data, origin='lower', cmap=cls['cmap'],
                   extent=[xmin, xmax, ymin, ymax],
                   vmin=0, vmax=vmax, aspect='auto')
    
    # State boundaries
    se.boundary.plot(ax=ax, color='#555555', linewidth=0.6, alpha=0.7)
    
    # City labels (disabled per user request)
    # for name, (clat, clon) in CITIES.items():
    #     ax.plot(clon, clat, 'o', color='white', markersize=4, markeredgecolor='#333', markeredgewidth=0.5)
    #     ax.text(clon, clat, f'  {name}', transform=ccrs.PlateCarree(),
    #             fontsize=8, fontweight='bold', color='#222',
    #             bbox=dict(boxstyle='round,pad=0.15', facecolor='white', 
    #                      edgecolor='#555', linewidth=0.5, alpha=0.85))
    
    # Colorbar
    cb = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label(f"{cls['label']} flow ({cls['unit']})" if abbr != 'Speed' else f"Speed ({cls['unit']})", 
                  fontsize=10)
    
    # Map features
    ax.coastlines(resolution='50m', linewidth=0.4, color='#555')
    ax.set_extent([-54, -39, -27, -14], crs=ccrs.PlateCarree())
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    title = f"{cls['label']} flow (veh/h)" if abbr != 'Speed' else f"Speed ({cls['unit']})"
    ax.set_title(f"{title} — Weekday 8am\n2D histogram",
                 fontweight='bold', fontsize=11)
    
    ax.gridlines(draw_labels=False, linewidth=0.3, color='grey', alpha=0.3)
    
    plt.tight_layout()
    path = f"{FIGS_DIR}/fig_gridded_{abbr.lower()}.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  {cls['label']:8s}: saved ({os.path.getsize(path)/1024:.0f} KB)")
    return fig, ax

for abbr, cls in CLASSES.items():
    if abbr not in grids: continue
    plot_class(abbr, grids[abbr], cls)

# ── 6. Combined 6-panel ──
print("\n[6] Generating combined 6-panel figure...")
fig, axes = plt.subplots(3, 2, figsize=(14, 18),
                         subplot_kw={'projection': ccrs.PlateCarree()})
used_axes = set()

for idx, (abbr, cls) in enumerate(CLASSES.items()):
    if abbr not in grids: continue
    row, col = divmod(idx, 2)
    ax = axes[row, col]
    used_axes.add((row, col))
    
    vmax = cls['vmax']
    data = grids[abbr].copy()
    data[data > vmax] = vmax
    
    im = ax.imshow(data, origin='lower', cmap=cls['cmap'],
                   extent=[xmin, xmax, ymin, ymax],
                   vmin=0, vmax=vmax, aspect='auto')
    
    se.boundary.plot(ax=ax, color='#555555', linewidth=0.4, alpha=0.6)
    
    # City labels (disabled)
    # for name, (clat, clon) in CITIES.items():
    #     ax.plot(clon, clat, 'o', color='white', markersize=2.5, markeredgecolor='#333', markeredgewidth=0.3)
    #     ax.text(clon, clat, f'  {name}', transform=ccrs.PlateCarree(),
    #             fontsize=6, fontweight='bold', color='#222',
    #             bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
    #                      edgecolor='#555', linewidth=0.3, alpha=0.85))
    
    ax.coastlines(resolution='50m', linewidth=0.3, color='#555')
    ax.set_extent([-54, -39, -27, -14], crs=ccrs.PlateCarree())
    ax.set_xlabel('Longitude', fontsize=7)
    ax.set_ylabel('Latitude', fontsize=7)
    ax.tick_params(labelsize=7)
    
    title_name = f"{cls['label']} flow" if abbr != 'Speed' else "Speed"
    ax.set_title(f"{title_name}\nMean {predictions[abbr].mean():.0f} | Max {vmax}",
                 fontweight='bold', fontsize=8)
    
    cb = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02, aspect=20)
    cb.set_label(cls['unit'], fontsize=6)
    cb.ax.tick_params(labelsize=5)

# Hide panels left empty because a model was unavailable (e.g. Speed retraining)
for row in range(3):
    for col in range(2):
        if (row, col) not in used_axes:
            axes[row, col].axis('off')

fig.suptitle('Predicted Traffic Flow & Speed — Weekday 8am\n2D Histogram Grid (~3 km), XGBoost SP Model',
             fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout()
path = f"{FIGS_DIR}/fig09_gridded_maps.png"
plt.savefig(path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  Combined: saved ({os.path.getsize(path)/1024:.0f} KB)")

print("\nDone. All maps generated.")
