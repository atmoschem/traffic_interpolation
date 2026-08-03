#!/usr/bin/env python3
"""
Step 2-3: Retrain all models with SP+RJ data, generate paper-quality figures.
Figures organized in xgboost_june_2026/figs/ with paper-quality specs:
  300 DPI, 10pt fonts, clean design, colorblind-friendly.
"""
import os, sys, json, warnings, time, gc
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.ticker as mticker

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
print("Step 2-3: Retrain + Paper Figures (SP+RJ) (Organized)")
print("="*60)

# ── 1. Load ALL training data ──
print("\n[1/6] Loading training data...")
t0 = time.time()
sp = pd.read_csv(SP_DATA)
print(f"  SP data: {len(sp):,} rows")

if os.path.exists(RJ_DATA):
    rj = pd.read_csv(RJ_DATA)
    print(f"  RJ data: {len(rj):,} rows")
    # Ensure column alignment
    missing = [c for c in sp.columns if c not in rj.columns and c != 'speed']
    for c in missing:
        rj[c] = np.nan
    common = ['speed'] + [c for c in sp.columns if c in rj.columns]
    common = list(dict.fromkeys(common))  # deduplicate
    full = pd.concat([sp, rj], ignore_index=True)
    print(f"  Combined: {len(full):,} rows")
else:
    print(f"  RJ data not found at {RJ_DATA}, using SP only")
    full = sp.copy()

print(f"  Time: {time.time()-t0:.0f}s")

# ── 2. Winsorize targets ──
print("\n[2/6] Winsorizing targets (p99 clip)...")
for t in TARGETS:
    if t in full.columns:
        upper = full[t].quantile(0.99)
        full[t] = full[t].clip(upper=upper)
        print(f"  {t}: p99={upper:.0f}")

# ── 3. Train models ──
print("\n[3/6] Training models...")
models = {}
train_results = {}

for target in TARGETS:
    if target not in full.columns:
        print(f"  {target}: not in data, skipping")
        continue
    cls = CLASSES[target]
    
    df_train = full.dropna(subset=[target]).copy()
    for c in STATIC + TEMPORAL:
        if c in df_train.columns:
            df_train[c] = df_train[c].fillna(FILL_VALS.get(c, 0.0))
    
    X = df_train[FEATURES]
    y = df_train[target].values
    
    # Temporal split: Mon-Wed train, Thu-Sun test
    train_mask = df_train['day_of_week'].isin([0,1,2])  # Mon, Tue, Wed
    test_mask = df_train['day_of_week'].isin([3,4,5,6])  # Thu-Sun
    
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
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    y_pred_train = model.predict(dtrain)
    r2_train = r2_score(y_train, y_pred_train)
    
    models[target] = model
    train_results[target] = {
        'r2': round(r2, 4), 'mae': round(mae, 2), 'rmse': round(rmse, 2),
        'r2_train': round(r2_train, 4), 'rounds': best_round,
        'n_train': len(y_train), 'n_test': len(y_test),
        'mean_obs': round(float(y_test.mean()), 1),
        'mean_pred': round(float(y_pred.mean()), 1),
        'evals': {'train_rmse': evals_result['train']['rmse'],
                  'test_rmse': evals_result['test']['rmse']}
    }
    
    model_path = f'{MODEL_DIR}/xgb_flow_{cls["short"]}_sprj.json'
    model.save_model(model_path)
    
    print(f"\n  {cls['label']} ({target}):")
    print(f"    R² = {r2:.4f} (train: {r2_train:.4f})")
    print(f"    MAE = {mae:.1f}, RMSE = {rmse:.1f}")
    print(f"    Best round: {best_round}")
    print(f"    Model: {model_path}")

with open(f'{MODEL_DIR}/results_sprj.json','w') as f:
    json.dump(train_results, f, indent=2)

# ── 4. Figure 1: Learning Curves ──
print("\n[4/6] Figure 1: Learning curves...")
fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))
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
    ax.set_title(f'{cls["label"]}', fontweight='bold')
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, len(tr))

fig.suptitle('Learning Curves — XGBoost Retrained with SP + RJ Data', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
save_fig('fig01_learning_curves.png', fig)

# ── 5. Figure 2: Validation Scatter ──
print("\n[5/6] Figure 2: Validation scatter...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, target in enumerate(TARGETS):
    if target not in models:
        continue
    cls = CLASSES[target]
    model = models[target]
    
    df_val = full.dropna(subset=[target]).copy()
    for c in FEATURES:
        df_val[c] = df_val[c].fillna(FILL_VALS.get(c, 0.0))
    
    X_val = df_val[FEATURES]
    y_obs = df_val[target].values
    y_pred = model.predict(xgb.DMatrix(X_val))
    
    ax = axes[idx]
    hb = ax.hexbin(y_obs, y_pred, gridsize=45, cmap='Blues', mincnt=1, alpha=0.8)
    lims = [min(y_obs.min(), y_pred.min()), max(y_obs.max(), y_pred.max())]
    ax.plot(lims, lims, 'r--', alpha=0.4, lw=1)
    
    r2 = r2_score(y_obs, y_pred)
    mae = mean_absolute_error(y_obs, y_pred)
    
    ax.set_xlabel(f'Observed {cls["label"]} flow (veh/h)')
    ax.set_ylabel(f'Predicted {cls["label"]} flow (veh/h)')
    ax.set_title(f'{cls["label"]}: R²={r2:.3f}, MAE={mae:.0f}', fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.colorbar(hb, ax=ax, label='Count')

for idx in range(len(TARGETS), 6):
    axes[idx].set_visible(False)

fig.suptitle('SP + RJ Model Validation (Mon-Wed train, Thu-Sun test)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
save_fig('fig02_validation_scatter.png', fig)

# ── 6. Figure 3: Validation by Day of Week ──
print("\n[5/6] Figure 3: R² by day of week...")
dow_names = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
fig, ax = plt.subplots(figsize=(8, 5))

x = np.arange(7)
width = 0.15
for idx, target in enumerate(TARGETS):
    if target not in models:
        continue
    cls = CLASSES[target]
    model = models[target]
    
    df_val = full.dropna(subset=[target]).copy()
    for c in FEATURES:
        df_val[c] = df_val[c].fillna(FILL_VALS.get(c, 0.0))
    
    r2_dow = []
    for dow in range(7):
        mask = df_val['day_of_week'] == dow
        if mask.sum() > 10:
            y_o = df_val[mask][target].values
            y_p = model.predict(xgb.DMatrix(df_val[mask][FEATURES]))
            r2_dow.append(round(r2_score(y_o, y_p), 3))
        else:
            r2_dow.append(np.nan)
    
    ax.bar(x + idx*width, r2_dow, width, label=cls['label'], color=cls['color'], alpha=0.85)

ax.set_xlabel('Day of week')
ax.set_ylabel('R²')
ax.set_title('Validation R² by Day of Week', fontweight='bold')
ax.set_xticks(x + width*2)
ax.set_xticklabels(dow_names)
ax.legend(loc='lower left', fontsize=8)
ax.axhline(0, color='gray', ls='-', alpha=0.3)
ax.set_ylim(-0.5, 1.0)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
save_fig('fig03_validation_dow.png', fig)

# ── 7. Figure 4: Feature Importance ──
print("\n[5/6] Figure 4: Feature importance...")
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
    ax.set_title(f'{cls["label"]}', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

for idx in range(len(TARGETS), 5):
    axes[idx].set_visible(False)

fig.suptitle('Feature Importance (Gain) — SP+RJ Models', fontsize=14, fontweight='bold')
plt.tight_layout()
save_fig('fig04_feature_importance.png', fig)

print("\n" + "="*60)
print("Retraining completed successfully.")
print("="*60)
