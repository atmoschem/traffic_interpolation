#!/usr/bin/env python3
"""
Step 6: Regenerate Figure 1 (flow validation scatter) with p-value annotations.
Loads pre-trained SP-only models, so no retraining is needed.
"""
import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

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

STATIC = [
    'lanes','maxspeed','nightlight','pop_density','road_density_500m',
    'road_density_1000m','road_density_5000m','highway_code',
    'surface_code','oneway_code','lu_tree','lu_shrubland','lu_grassland',
    'lu_cropland','lu_builtup','lu_bare','lu_water','lu_wetland',
    'lu_mangroves','lu_moss'
]
TEMPORAL = ['hour','day_of_week','is_weekend']
FEATURES = STATIC + TEMPORAL

CLASSES = {
    'flow_pc':    {'label': 'PC',    'color': '#2166AC', 'short': 'pc'},
    'flow_lcv':   {'label': 'LCV',   'color': '#D6604D', 'short': 'lcv'},
    'flow_mc':    {'label': 'MC',    'color': '#4DAF4A', 'short': 'mc'},
    'flow_truck': {'label': 'Truck', 'color': '#984EA3', 'short': 'truck'},
    'flow_bus':   {'label': 'Bus',   'color': '#FF7F00', 'short': 'bus'}
}
TARGETS = list(CLASSES.keys())

FILL_VALS = {
    'lanes': 2.0, 'maxspeed': 60.0, 'nightlight': 0.0, 'pop_density': 0.0,
    'road_density_500m': 0.0, 'road_density_1000m': 0.0, 'road_density_5000m': 0.0,
    'highway_code': 4.0, 'surface_code': 1.0, 'oneway_code': 1.0
}
for c in STATIC:
    if c.startswith('lu_'):
        FILL_VALS[c] = 0.0


def safe_fillna(df, features, fill_vals):
    for c in features:
        if c in df.columns:
            df[c] = df[c].fillna(fill_vals.get(c, 0.0))


def p_value_label(y_true, y_pred):
    if len(y_true) < 3:
        return 'p=n/a'
    r, p = pearsonr(y_true, y_pred)
    if p < 0.001:
        return 'p<0.001', r, p
    return f'p={p:.3f}', r, p


print('=' * 60)
print('Step 6: Regenerate Figure 1 (flow validation scatter, SP-only models)')
print('=' * 60)

print('Loading training data...')
sp = pd.read_csv(SP_DATA)
print(f'  SP data: {len(sp):,} rows')

print('Filling missing features...')
safe_fillna(sp, FEATURES, FILL_VALS)

print('Selecting test set: Thu-Sun...')
df_test = sp[sp['day_of_week'].isin([3, 4, 5, 6])].copy()
print(f'  Test set: {len(df_test):,} rows')

records = []

for target in TARGETS:
    cls = CLASSES[target]
    # Prefer Python-native SP-only model; fall back to the R-saved one.
    model_path = os.path.join(MODEL_DIR, f'xgb_flow_{cls["short"]}_sp.json')
    if not os.path.exists(model_path):
        model_path = os.path.join(MODEL_DIR, f'xgb_flow_{cls["short"]}_sp_r.json')
    if not os.path.exists(model_path):
        print(f'  Model not found: {model_path} — skipping')
        continue

    model = xgb.Booster()
    model.load_model(model_path)
    # R-saved models store best_iteration; slice trees to match R predict() behavior
    best_iter = model.attr('best_iteration')
    iteration_range = (0, int(best_iter) + 1) if best_iter is not None else (0, 0)

    y_obs = df_test[target].to_numpy()
    X_test = df_test[FEATURES].to_numpy()
    dtest = xgb.DMatrix(X_test, feature_names=FEATURES)
    y_pred = model.predict(dtest, iteration_range=iteration_range)

    valid = (~np.isnan(y_obs)) & np.isfinite(y_pred)
    y_obs = y_obs[valid]
    y_pred = y_pred[valid]

    if len(y_obs) == 0:
        print(f'  {cls["label"]}: no valid rows, skipping')
        continue

    r2_v = r2_score(y_obs, y_pred)
    mae_v = mean_absolute_error(y_obs, y_pred)
    pval_str, r_val, p_val = p_value_label(y_obs, y_pred)
    print(f'  {cls["label"]}: R²={r2_v:.3f}, MAE={mae_v:.0f}, {pval_str} (r={r_val:.3f}, n={len(y_obs)})')

    lims = [min(y_obs.min(), y_pred.min()), max(y_obs.max(), y_pred.max())]
    use_hex = (lims[1] - lims[0]) > 1.0

    records.append({
        'label': cls['label'],
        'color': cls['color'],
        'y_obs': y_obs,
        'y_pred': y_pred,
        'r2': r2_v,
        'mae': mae_v,
        'pval_str': pval_str,
        'lims': lims,
        'use_hex': use_hex
    })

if len(records) == 0:
    print('No plots were generated because no models were found.')
    sys.exit(1)

selected_records = records[:4]
cols = 2
rows = (len(selected_records) + cols - 1) // cols
fig_final, axes = plt.subplots(rows, cols, figsize=(13, 12), squeeze=False)
axes = axes.flatten()

for idx, rec in enumerate(selected_records):
    ax = axes[idx]
    if rec['use_hex']:
        hb = ax.hexbin(rec['y_obs'], rec['y_pred'], gridsize=45, cmap='Blues',
                       norm=LogNorm(), mincnt=1, alpha=0.8)
        cb = fig_final.colorbar(hb, ax=ax)
        cb.set_label('Count')
    else:
        ax.scatter(rec['y_obs'], rec['y_pred'], alpha=0.4, s=10, color=rec['color'])

    ax.plot(rec['lims'], rec['lims'], 'r--', alpha=0.4, linewidth=1.0)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(rec['lims'])
    ax.set_ylim(rec['lims'])
    ax.set_xlabel(f'Observed {rec["label"]} flow (veh/h)')
    ax.set_ylabel(f'Predicted {rec["label"]} flow (veh/h)')
    ax.set_title(f'{rec["label"]}: R²={rec["r2"]:.3f}, MAE={rec["mae"]:.0f}, {rec["pval_str"]}', fontsize=10)
    ax.grid(alpha=0.3)

for ax in axes[len(selected_records):]:
    ax.set_visible(False)

fig_final.suptitle('Flow Model Validation — SP-only (Test set: Thu–Sun)', fontsize=16, fontweight='bold')
fig_path = os.path.join(FIGS_DIR, 'fig01_flow_validation_scatter_pval.png')
fig_final.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close(fig_final)
print(f'\nSaved: {fig_path}')
print('Done.')
