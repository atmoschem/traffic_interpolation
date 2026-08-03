#!/usr/bin/env python3
"""
Step 3: Proper RJ validation with temporal holdout.
Only evaluates on RJ data NOT seen during training (Thu-Sun).
"""
import os, warnings, json
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({'font.family':'sans-serif','font.size':10,'axes.titlesize':12,
    'axes.labelsize':11,'savefig.dpi':300,'savefig.bbox':'tight',
    'axes.grid':True,'grid.alpha':0.3})

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

MODEL_DIR = f"{BASE}/xgboost_june_2026/models"
RJ_TRAIN = f"{BASE}/xgboost_june_2026/cetrio_training_data.csv"
SP_DATA = f"{BASE}/xgboost_june_2026/training_data_fixed_rd.csv"
FIGS_DIR = f"{BASE}/xgboost_june_2026/final/pyfigs"

STATIC = ['lanes','maxspeed','nightlight','pop_density','road_density_500m',
          'road_density_1000m','road_density_5000m','highway_code',
          'surface_code','oneway_code','lu_tree','lu_shrubland','lu_grassland',
          'lu_cropland','lu_builtup','lu_bare','lu_water','lu_wetland',
          'lu_mangroves','lu_moss']
FEATURES = STATIC + ['hour','day_of_week','is_weekend']
FILL_VALS = {'lanes':2.0,'maxspeed':60.0,'nightlight':0.0,'pop_density':0.0,
             'road_density_500m':0.0,'road_density_1000m':0.0,'road_density_5000m':0.0,
             'highway_code':4.0,'surface_code':1.0,'oneway_code':1.0}
for c in STATIC:
    if c.startswith('lu_'): FILL_VALS[c] = 0.0

CLASSES = {'flow_pc':('PC','#2166AC','pc'),'flow_lcv':('LCV','#D6604D','lcv'),
           'flow_mc':('MC','#4DAF4A','mc'),'flow_truck':('Truck','#984EA3','truck')}
TARGETS = list(CLASSES.keys())

print("="*60)
print("CET-Rio Proper Validation (Thu-Sun test only)")
print("="*60)

# Load data, separate test RJ
rj = pd.read_csv(RJ_TRAIN)
sp = pd.read_csv(SP_DATA)

# RJ test set: Thu-Sun (never seen by model)
rj_test = rj[rj['day_of_week'].isin([3,4,5,6])].copy()
rj_train = rj[rj['day_of_week'].isin([0,1,2])].copy()
print(f"\n  RJ total: {len(rj):,}")
print(f"  RJ test (Thu-Sun): {len(rj_test):,}")
print(f"  RJ train (Mon-Wed): {len(rj_train):,}")

# SP-only model predictions for comparison (same RJ test data)
print("\n\nSP-only model (baseline) on RJ test data:")
sp_results = {}
for target in TARGETS:
    if target == 'flow_bus':
        continue
    cls_name, color, short = CLASSES[target]
    
    # Try finding the SP-only baseline model
    sp_model_path = f"{BASE}/models/xgb_flow_{short}.json"
    if not os.path.exists(sp_model_path):
        # Fallback check
        sp_model_path = f"{BASE}/xgboost_june_2026/models/xgb_flow_{short}_sp_r.json"
        
    if os.path.exists(sp_model_path) or os.path.exists(sp_model_path.replace('_sp_r', '')):
        sp_model = xgb.XGBRegressor()
        if os.path.exists(sp_model_path):
            sp_model.load_model(sp_model_path)
        else:
            sp_model.load_model(sp_model_path.replace('_sp_r', ''))
        
        df_v = rj_test.copy()
        for c in FEATURES:
            if c in df_v.columns:
                df_v[c] = df_v[c].fillna(FILL_VALS.get(c, 0.0))
        
        y_o = df_v[target].values
        dm = xgb.DMatrix(df_v[FEATURES], feature_names=FEATURES)
        y_p = sp_model.get_booster().predict(dm)
        
        r2 = r2_score(y_o, y_p)
        mae = mean_absolute_error(y_o, y_p)
        bias = y_p.mean()/y_o.mean() if y_o.mean()>0 else np.nan
        sp_results[target] = {'r2':round(r2,4),'bias':round(float(bias),3),
                              'mean_obs':round(float(y_o.mean()),1),
                              'mean_pred':round(float(y_p.mean()),1)}
        print(f"  {cls_name:>6}: R²={r2:.4f} bias={bias:.2f}x obs={y_o.mean():.1f} pred={y_p.mean():.1f}")
    else:
        print(f"  {cls_name:>6}: SP baseline model not found at {sp_model_path}")
        sp_results[target] = {'r2':np.nan,'bias':np.nan,'mean_obs':np.nan,'mean_pred':np.nan}

# SP+RJ model on same RJ test data
print("\n\nSP+RJ model on RJ test data:")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
results = {}

for idx, target in enumerate(TARGETS):
    cls_name, color, short = CLASSES[target]
    model_path = f'{MODEL_DIR}/xgb_flow_{short}_sprj.json'
    if not os.path.exists(model_path):
        print(f"  {cls_name:>6}: SP+RJ model not found at {model_path}")
        continue
    
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    df_v = rj_test.copy()
    for c in FEATURES:
        if c in df_v.columns:
            df_v[c] = df_v[c].fillna(FILL_VALS.get(c, 0.0))
    
    y_o = df_v[target].values
    dm = xgb.DMatrix(df_v[FEATURES], feature_names=FEATURES)
    y_p = model.get_booster().predict(dm)
    
    r2 = r2_score(y_o, y_p)
    mae = mean_absolute_error(y_o, y_p)
    rmse = np.sqrt(mean_squared_error(y_o, y_p))
    bias = y_p.mean()/y_o.mean() if y_o.mean()>0 else np.nan
    
    results[target] = {'r2':round(r2,4),'mae':round(mae,2),'rmse':round(rmse,2),
                       'mean_obs':round(float(y_o.mean()),1),
                       'mean_pred':round(float(y_p.mean()),1),
                       'bias_ratio':round(float(bias),3),'n':len(y_o)}
    
    old_bias = sp_results.get(target,{}).get('bias', 'N/A')
    print(f"  {cls_name:>6}: R²={r2:.4f} bias={bias:.2f}x (was {old_bias}) "
          f"obs={y_o.mean():.1f} pred={y_p.mean():.1f}")
    
    ax = axes[idx]
    hb = ax.hexbin(y_o, y_p, gridsize=40, cmap='Blues', mincnt=1, alpha=0.8)
    lims = [0, max(y_o.max(), y_p.max())]
    ax.plot(lims, lims, 'r--', alpha=0.4, lw=1)
    ax.set_xlabel(f'Observed {cls_name} (veh/h)')
    ax.set_ylabel(f'Predicted {cls_name} (veh/h)')
    ax.set_title(f'{cls_name}: R²={r2:.3f}, bias={bias:.2f}x', fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.colorbar(hb, ax=ax, label='Count')

fig.suptitle('SP+RJ Model → RJ Test (Thu-Sun, held-out)', fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(f'{FIGS_DIR}/fig06_rj_validation_temporal.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"\n  Figure: {FIGS_DIR}/fig06_rj_validation_temporal.png")

# Summary
print("\n" + "="*60)
print("Bias improvement: SP-only → SP+RJ")
print(f"{'Class':<10} {'SP-only bias':<15} {'SP+RJ bias':<15} {'R²':<8}")
print("-"*48)
for target in TARGETS:
    if target in results and target in sp_results:
        s = sp_results[target]
        r = results[target]
        print(f"{CLASSES[target][0]:<10} {s['bias']:<15} {r['bias_ratio']:<15.3f} {r['r2']:<8.4f}")

with open(f'{MODEL_DIR}/rj_validation_proper.json','w') as f:
    json.dump(results, f, indent=2)
print(f"Done. Results: {MODEL_DIR}/rj_validation_proper.json")
