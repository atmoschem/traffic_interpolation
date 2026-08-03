#!/usr/bin/env python3
"""
Step 8: Error histograms + Shapiro-Wilk normality test for each flow model.
Manuscript Figure 2: Prediction Error Distributions — Flow Models (Thu-Sun test set).
Uses pre-trained SP-only models on the fixed road-density SP data
(python xgb_flow_{class}_sp.json, falling back to the R-saved _sp_r.json).
"""
import os, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import shapiro, norm
from sklearn.metrics import r2_score, mean_absolute_error

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ── Paper-quality styling ──
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['DejaVu Sans']
rcParams['font.size'] = 10
rcParams['axes.titlesize'] = 12
rcParams['axes.labelsize'] = 11
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 8
rcParams['figure.dpi'] = 150
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'
rcParams['axes.grid'] = True
rcParams['grid.alpha'] = 0.3
rcParams['axes.axisbelow'] = True

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

SP_DATA = f"{BASE}/xgboost_june_2026/training_data_fixed_rd.csv"
MODEL_DIR = f"{BASE}/xgboost_june_2026/models"
FIGS_DIR = f"{BASE}/xgboost_june_2026/final/pyfigs"
os.makedirs(FIGS_DIR, exist_ok=True)

STATIC = ['lanes','maxspeed','nightlight','pop_density','road_density_500m',
          'road_density_1000m','road_density_5000m','highway_code',
          'surface_code','oneway_code','lu_tree','lu_shrubland','lu_grassland',
          'lu_cropland','lu_builtup','lu_bare','lu_water','lu_wetland',
          'lu_mangroves','lu_moss']
TEMPORAL = ['hour','day_of_week','is_weekend']
FEATURES = STATIC + TEMPORAL  # 23

CLASSES = {
    'flow_pc':   {'label':'PC',    'color':'#2166AC', 'short':'pc'},
    'flow_lcv':  {'label':'LCV',   'color':'#D6604D', 'short':'lcv'},
    'flow_mc':   {'label':'MC',    'color':'#4DAF4A', 'short':'mc'},
    'flow_truck':{'label':'Truck', 'color':'#984EA3', 'short':'truck'}
}
TARGETS = list(CLASSES.keys())

FILL_VALS = {'lanes':2.0,'maxspeed':60.0,'nightlight':0.0,'pop_density':0.0,
             'road_density_500m':0.0,'road_density_1000m':0.0,'road_density_5000m':0.0,
             'highway_code':4.0,'surface_code':1.0,'oneway_code':1.0}
for c in STATIC:
    if c.startswith('lu_'): FILL_VALS[c] = 0.0

print("="*60)
print("Step 8: Flow Error Distributions (SP-only, Thu-Sun test set)")
print("="*60)

# ── 1. Load data, Thu-Sun test set ──
print("\n[1] Loading data...")
sp = pd.read_csv(SP_DATA)
df_test = sp[sp['day_of_week'].isin([3, 4, 5, 6])].copy()
for c in FEATURES:
    if c in df_test.columns:
        df_test[c] = df_test[c].fillna(FILL_VALS.get(c, 0.0))
print(f"  Test set: {len(df_test):,} rows")

X_test = df_test[FEATURES].values.astype(np.float32)
dm_test = xgb.DMatrix(X_test, feature_names=FEATURES)

# ── 2. Per-class residuals + Shapiro-Wilk ──
shapiro_results = []
panels = {}

for target in TARGETS:
    cls = CLASSES[target]
    model_path = f"{MODEL_DIR}/xgb_flow_{cls['short']}_sp.json"
    if not os.path.exists(model_path):
        model_path = f"{MODEL_DIR}/xgb_flow_{cls['short']}_sp_r.json"
    if not os.path.exists(model_path):
        print(f"  Model not found for {cls['label']} — skipping")
        continue

    model = xgb.Booster()
    model.load_model(model_path)

    y_obs = df_test[target].values
    # R's predict() honors the stored best_iteration from early stopping;
    # replicate that so R-saved models evaluate identically in both pipelines.
    if model.best_iteration is not None:
        y_pred = model.predict(dm_test, iteration_range=(0, model.best_iteration + 1))
    else:
        y_pred = model.predict(dm_test)

    valid = ~np.isnan(y_obs) & np.isfinite(y_pred)
    y_obs = y_obs[valid]
    y_pred = y_pred[valid]

    residuals = y_obs - y_pred
    r2 = r2_score(y_obs, y_pred)
    mae = mean_absolute_error(y_obs, y_pred)
    rmse = np.sqrt(np.mean(residuals**2))
    mu = residuals.mean()
    sd = residuals.std(ddof=1)  # match R sd()

    # Shapiro-Wilk test (max 5000 samples, matching R limitation)
    n_shap = min(5000, len(residuals))
    if len(residuals) > 5000:
        rng = np.random.RandomState(42)
        idx_shap = rng.choice(len(residuals), n_shap, replace=False)
    else:
        idx_shap = np.arange(len(residuals))
    sw_stat, sw_p = shapiro(residuals[idx_shap])

    skewness = np.mean((residuals - mu)**3) / (sd**3)  # match R formula

    print(f"\n{cls['label']}:")
    print(f"  R²={r2:.3f}, MAE={mae:.0f}, RMSE={rmse:.0f}")
    print(f"  Residuals: mean={mu:.1f}, sd={sd:.0f}, n={len(residuals)}")
    print(f"  Shapiro-Wilk: W={sw_stat:.4f}, p={sw_p:.2e}")

    shapiro_results.append({
        'class': cls['label'],
        'n': len(residuals),
        'mean_resid': round(mu, 1),
        'sd_resid': round(sd, 0),
        'skewness': round(skewness, 3),
        'shapiro_W': round(sw_stat, 4),
        'shapiro_p': sw_p
    })
    panels[target] = (residuals, r2, mae, sw_stat, sw_p)

# ── 3. Figure: 2x2 histogram panels with normal curve overlay ──
print("\n[2] Generating figure...")
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

props = dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85,
             edgecolor='gray', linewidth=0.5)

for idx, target in enumerate(TARGETS):
    if target not in panels:
        axes[idx].set_visible(False)
        continue
    cls = CLASSES[target]
    residuals, r2, mae, sw_stat, sw_p = panels[target]

    ax = axes[idx]
    ax.hist(residuals, bins=80, density=True, alpha=0.7,
            color=cls['color'], edgecolor='white', linewidth=0.3)

    mu, sigma = residuals.mean(), residuals.std(ddof=1)
    x_grid = np.linspace(residuals.min(), residuals.max(), 500)
    ax.plot(x_grid, norm.pdf(x_grid, mu, sigma),
            color='black', linewidth=1.5, linestyle='--')

    sw_label = f"Shapiro-Wilk\nW = {sw_stat:.4f}\np = {sw_p:.2e}"
    ax.text(0.97, 0.97, sw_label, transform=ax.transAxes,
            fontsize=9, fontweight='bold', ha='right', va='top', bbox=props)

    ax.set_xlabel('Residual (observed - predicted, veh/h)')
    ax.set_ylabel('Density')
    ax.set_title(f"{cls['label']} — Residuals (R²={r2:.3f}, MAE={mae:.0f})",
                 fontweight='bold')
    ax.grid(True, alpha=0.3)

fig.suptitle('Prediction Error Distributions — Flow Models (Thu-Sun test set)',
             fontsize=18, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig_path = f'{FIGS_DIR}/fig07_flow_error_distributions.png'
fig.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved: {fig_path}")

# ── 4. Shapiro results table ──
print("\n\n=== Normality Test Summary ===")
shapiro_df = pd.DataFrame(shapiro_results)
print(shapiro_df.to_string(index=False))
csv_path = f'{FIGS_DIR}/fig07_shapiro_results.csv'
shapiro_df.to_csv(csv_path, index=False)
print(f"Saved: {csv_path}")
print("Done.")
