#!/usr/bin/env python3
"""
Step 0: Train SP-only flow models (manuscript Table 1), validate per-class on SP
test set + total flow on RJ. Python mirror of final/R/00_train_sp_only.R.
Figures organized in xgboost_june_2026/final/pyfigs/ with paper-quality specs:
  300 DPI, 10pt fonts, clean design, colorblind-friendly.

Strategy:
  Train:  SP-only per-class flow (Mon-Wed)
  Validate: SP test set (Thu-Sun) per-class
  Cross-validate: Predict on RJ, sum 4 classes -> compare vs observed RJ total
"""
import os, sys, json, warnings, time
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

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
RJ_DATA = f"{BASE}/xgboost_june_2026/cetrio_training_data.csv"
MODEL_DIR = f"{BASE}/xgboost_june_2026/models"
FIGS_DIR = f"{BASE}/xgboost_june_2026/final/pyfigs"
os.makedirs(MODEL_DIR, exist_ok=True)
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
    'flow_truck':{'label':'Truck', 'color':'#984EA3', 'short':'truck'},
    'flow_bus':  {'label':'Bus',   'color':'#FF7F00', 'short':'bus'}
}
TARGETS = list(CLASSES.keys())
TRAIN_PARAMS = {
    'objective':'reg:squarederror','max_depth':8,'learning_rate':0.08,
    'subsample':0.8,'colsample_bytree':0.8,'min_child_weight':5,
    'gamma':1,'seed':42,'n_jobs':-1
}

FILL_VALS = {'lanes':2.0,'maxspeed':60.0,'nightlight':0.0,'pop_density':0.0,
             'road_density_500m':0.0,'road_density_1000m':0.0,'road_density_5000m':0.0,
             'highway_code':4.0,'surface_code':1.0,'oneway_code':1.0}
for c in STATIC:
    if c.startswith('lu_'): FILL_VALS[c] = 0.0

def save_fig(name, fig=None):
    path = f'{FIGS_DIR}/{name}'
    if fig is None:
        fig = plt.gcf()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path

print("="*60)
print("Flow Models: SP-only train, per-class + RJ total validation")
print("="*60)

# ── 1. Load data ──
print("\n[1/7] Loading data...")
t0 = time.time()
sp = pd.read_csv(SP_DATA)
rj = pd.read_csv(RJ_DATA)
print(f"  SP: {len(sp):,} rows | RJ: {len(rj):,} rows")
print(f"  Time: {time.time()-t0:.0f}s")

# ── 2. Train SP-only flow models ──
print("\n[2/7] Training SP-only flow models (Mon-Wed train, Thu-Sun test)...")
models = {}
train_results = {}

for target in TARGETS:
    if target not in sp.columns:
        print(f"  {target}: not in SP data, skipping")
        continue
    cls = CLASSES[target]

    df = sp.dropna(subset=[target]).copy()
    for c in FEATURES:
        if c in df.columns:
            df[c] = df[c].fillna(FILL_VALS.get(c, 0.0))

    X = df[FEATURES]
    y = df[target].values

    # Mon-Wed (0,1,2) train, Thu-Sun (3,4,5,6) test
    train_mask = df['day_of_week'].isin([0,1,2])
    test_mask = df['day_of_week'].isin([3,4,5,6])

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    evals_result = {}
    model = xgb.train(
        TRAIN_PARAMS, dtrain, num_boost_round=500,
        evals=[(dtrain,'train'),(dtest,'test')],
        early_stopping_rounds=30,
        evals_result=evals_result,
        verbose_eval=False
    )

    best_round = model.best_iteration
    y_pred = model.predict(dtest)
    y_pred_train = model.predict(dtrain)

    r2 = r2_score(y_test, y_pred)
    r2_tr = r2_score(y_train, y_pred_train)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    models[target] = model
    train_results[target] = {
        'r2': round(r2, 4), 'mae': round(mae, 2), 'rmse': round(rmse, 2),
        'r2_train': round(r2_tr, 4), 'rounds': best_round,
        'n_train': len(y_train), 'n_test': len(y_test),
        'mean_obs': round(float(y_test.mean()), 1),
        'mean_pred': round(float(y_pred.mean()), 1),
        'evals': {'train_rmse': evals_result['train']['rmse'],
                  'test_rmse': evals_result['test']['rmse']}
    }

    model_path = f'{MODEL_DIR}/xgb_flow_{cls["short"]}_sp.json'
    model.save_model(model_path)

    print(f"  {cls['label']} (SP-only): R²={r2:.4f} (train: {r2_tr:.4f})  "
          f"MAE={mae:.1f}  RMSE={rmse:.1f}  rounds={best_round}")

with open(f'{MODEL_DIR}/results_sp_flow.json','w') as f:
    json.dump(train_results, f, indent=2)

# ── 3. Figure 1: Learning Curves (SP test set) ──
print("\n[3/7] Figure 1: Learning curves (SP test set)...")
fig, axes = plt.subplots(1, 5, figsize=(22, 5))
for idx, target in enumerate(TARGETS):
    if target not in models:
        continue
    cls = CLASSES[target]
    evals = train_results[target]['evals']
    best = train_results[target]['rounds']

    ax = axes[idx]
    tr = evals['train_rmse']
    te = evals['test_rmse']
    epochs = range(1, len(tr)+1)

    ax.plot(epochs, tr, color='#333333', alpha=0.5, lw=0.8, label='Train')
    ax.plot(epochs, te, color=cls['color'], lw=1.2, label='Test')
    ax.axvline(best, color='red', ls='--', alpha=0.4, lw=0.8)
    ax.text(best, max(te)*0.9, f'  n={best}', fontsize=7, color='red', va='top')

    ax.set_xlabel('Boosting round')
    ax.set_ylabel('RMSE (veh/h)')
    ax.set_title(cls['label'], fontweight='bold')
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, len(tr))

fig.suptitle('SP-only Flow Models — Learning Curves (Mon-Wed train, Thu-Sun test)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
save_fig('fig01_flow_learning_curves.png', fig)

# ── 4. Figure 2: Per-class Validation Scatter (SP test set only) ──
print("\n[4/7] Figure 2: Per-class validation (SP test set hexbin)...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, target in enumerate(TARGETS):
    if target not in models:
        continue
    cls = CLASSES[target]
    model = models[target]

    df = sp.dropna(subset=[target]).copy()
    for c in FEATURES:
        if c in df.columns:
            df[c] = df[c].fillna(FILL_VALS.get(c, 0.0))

    # Test set only (Thu-Sun)
    df_test = df[df['day_of_week'].isin([3,4,5,6])]

    y_obs = df_test[target].values
    y_pred = model.predict(xgb.DMatrix(df_test[FEATURES]))

    ax = axes[idx]
    hb = ax.hexbin(y_obs, y_pred, gridsize=45, cmap='Blues', mincnt=1, alpha=0.8)
    lims = [min(y_obs.min(), y_pred.min()), max(y_obs.max(), y_pred.max())]
    ax.plot(lims, lims, 'r--', alpha=0.4, lw=1)

    r2_v = r2_score(y_obs, y_pred)
    mae_v = mean_absolute_error(y_obs, y_pred)
    pval = pearsonr(y_obs, y_pred)[1]
    pval_str = 'p<0.001' if pval < 0.001 else f'p={pval:.3f}'

    ax.set_xlabel(f'Observed {cls["label"]} flow (veh/h)')
    ax.set_ylabel(f'Predicted {cls["label"]} flow (veh/h)')
    ax.set_title(f'{cls["label"]} (SP test set): R²={r2_v:.3f}, MAE={mae_v:.0f}, {pval_str}',
                 fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.colorbar(hb, ax=ax, label='Count')
    print(f"  {cls['label']}: R²={r2_v:.3f} MAE={mae_v:.0f} (n={len(y_obs):,})")

for idx in range(len(TARGETS), 6):
    axes[idx].set_visible(False)

fig.suptitle('SP-only Flow Models — Test Set Validation (Thu-Sun)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
save_fig('fig02_flow_validation_scatter.png', fig)

# ── 5. Figure 3: Feature Importance (SP models) ──
print("\n[5/7] Figure 3: Feature importance (SP models)...")
n_top = 12
fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))

for idx, target in enumerate(TARGETS):
    if target not in models:
        continue
    cls = CLASSES[target]
    model = models[target]

    importance = model.get_score(importance_type='gain')
    items = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:n_top]
    feat_names = [x[0] for x in items[::-1]]
    feat_vals = [x[1] for x in items[::-1]]

    ax = axes[idx]
    ax.barh(range(len(feat_names)), feat_vals, color=cls['color'], alpha=0.8, height=0.7)
    ax.set_yticks(range(len(feat_names)))
    ax.set_yticklabels(feat_names, fontsize=7)
    ax.set_xlabel('Gain')
    ax.set_title(cls['label'], fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

for idx in range(len(TARGETS), 5):
    axes[idx].set_visible(False)

fig.suptitle('Feature Importance (Gain) — SP-only Flow Models', fontsize=14, fontweight='bold')
plt.tight_layout()
save_fig('fig03_flow_importance.png', fig)

# ── 6. Figure 4: RJ Total Flow Validation ──
print("\n[6/7] Figure 4: RJ total flow validation...")
# Compute RJ observed total (sum of 4 estimated classes = original total from CET-Rio)
rj['rj_total_obs'] = rj['flow_pc'] + rj['flow_lcv'] + rj['flow_mc'] + rj['flow_truck']

# Fill NaN features on RJ data
for c in FEATURES:
    if c in rj.columns:
        rj[c] = rj[c].fillna(FILL_VALS.get(c, 0.0))
    else:
        rj[c] = FILL_VALS.get(c, 0.0)
X_rj = xgb.DMatrix(rj[FEATURES])

# Predict per-class; total predicted = sum of the 4 RJ classes (no bus in RJ)
rj_classes = ['flow_pc','flow_lcv','flow_mc','flow_truck']
rj['rj_total_pred'] = 0.0
for target in rj_classes:
    if target not in models:
        continue
    cls = CLASSES[target]
    rj[f'pred_{cls["short"]}'] = models[target].predict(X_rj)
    rj['rj_total_pred'] += rj[f'pred_{cls["short"]}']

valid = rj['rj_total_obs'].notna()
y_rj_obs = rj.loc[valid, 'rj_total_obs'].values
y_rj_pred = rj.loc[valid, 'rj_total_pred'].values

r2_rj = r2_score(y_rj_obs, y_rj_pred)
mae_rj = mean_absolute_error(y_rj_obs, y_rj_pred)
rmse_rj = np.sqrt(mean_squared_error(y_rj_obs, y_rj_pred))
bias_rj = y_rj_pred.mean() / y_rj_obs.mean()

print(f"  RJ total flow: R²={r2_rj:.4f}  MAE={mae_rj:.1f}  RMSE={rmse_rj:.1f}  bias={bias_rj:.3f}")

fig, ax = plt.subplots(figsize=(8, 8))
hb = ax.hexbin(y_rj_obs, y_rj_pred, gridsize=50, cmap='Blues', mincnt=1, alpha=0.8)
lims_rj = [min(y_rj_obs.min(), y_rj_pred.min()), max(y_rj_obs.max(), y_rj_pred.max())]
ax.plot(lims_rj, lims_rj, 'r--', alpha=0.4, lw=1)
ax.set_xlim(lims_rj)
ax.set_ylim(lims_rj)
ax.set_aspect('equal')
ax.set_xlabel('Observed total flow (veh/h) — CET-Rio')
ax.set_ylabel('Predicted total flow (veh/h) — sum of 4 classes')
ax.set_title(f'RJ Total Flow Validation: R²={r2_rj:.3f}, MAE={mae_rj:.0f}, bias={bias_rj:.2f}',
             fontweight='bold')
ax.grid(True, alpha=0.3)
plt.colorbar(hb, ax=ax, label='Count')
plt.tight_layout()
save_fig('fig04_rj_total_validation.png', fig)

# Also save RJ validation to results
rj_results = {
    'r2': round(r2_rj, 4), 'mae': round(mae_rj, 2), 'rmse': round(rmse_rj, 2),
    'bias': round(float(bias_rj), 4), 'n': int(len(y_rj_obs))
}
with open(f'{MODEL_DIR}/results_rj_validation.json','w') as f:
    json.dump(rj_results, f, indent=2)

# ── 7. Figure 5: R² by Day of Week (SP test set) ──
print("\n[7/7] Figure 5: R² by day of week (SP test set)...")
dow_names = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
test_days = [3,4,5,6]  # Thu-Sun
fig, ax = plt.subplots(figsize=(8, 5))

x = np.arange(len(test_days))
width = 0.15
for idx, target in enumerate(TARGETS):
    if target not in models:
        continue
    cls = CLASSES[target]
    model = models[target]

    df = sp.dropna(subset=[target]).copy()
    for c in FEATURES:
        if c in df.columns:
            df[c] = df[c].fillna(FILL_VALS.get(c, 0.0))

    r2_dow = []
    for d in test_days:
        mask = df['day_of_week'] == d
        if mask.sum() > 10:
            y_o = df.loc[mask, target].values
            y_p = model.predict(xgb.DMatrix(df.loc[mask, FEATURES]))
            r2_dow.append(round(r2_score(y_o, y_p), 3))
        else:
            r2_dow.append(np.nan)

    ax.bar(x + idx*width, r2_dow, width, label=cls['label'], color=cls['color'], alpha=0.85)

ax.set_xlabel('Day of week (test set)')
ax.set_ylabel('R²')
ax.set_title('SP-only Flow Models — Validation R² by Day of Week', fontweight='bold')
ax.set_xticks(x + width*2)
ax.set_xticklabels([dow_names[d] for d in test_days])
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=5, fontsize=8)
ax.axhline(0, color='gray', ls='-', alpha=0.3)
ax.set_ylim(-0.5, 1.0)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
save_fig('fig05_flow_dow_validation.png', fig)

# ── Final summary ──
print("\n" + "="*60)
print("SP-only Flow Model Summary")
print("="*60)
for target in TARGETS:
    if target not in train_results:
        continue
    r = train_results[target]
    print(f"  {CLASSES[target]['label']:>6s}: R²={r['r2']:.3f}  MAE={r['mae']:.1f}  "
          f"RMSE={r['rmse']:.1f}  rounds={r['rounds']}")
print(f"\n  RJ total validation: R²={r2_rj:.3f}  MAE={mae_rj:.1f}  bias={bias_rj:.2f}")

print(f"\nModels saved to: {MODEL_DIR}/")
print(f"Figures saved to: {FIGS_DIR}/")
print("Done.")
