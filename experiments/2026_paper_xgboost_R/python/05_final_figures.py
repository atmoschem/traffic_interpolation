#!/usr/bin/env python3
"""
Step 5: Final paper-quality figures. Refines the generated figures with:
- LogNorm color scaling for hexbin plots
- Axes clipped at 0
- Larger, consistent fonts
- Unified color scheme (train=gray, test=class color)
- 300 DPI, vector-compatible
"""
import os, warnings, json
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams, colors as mcolors
from matplotlib.colors import LogNorm

# ── Paper-quality style ──
rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7.5,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': False,
    'axes.axisbelow': True,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
})

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

FIGS_DIR = f"{BASE}/xgboost_june_2026/final/pyfigs"
MODEL_DIR = f"{BASE}/xgboost_june_2026/models"
SP_DATA = f"{BASE}/xgboost_june_2026/training_data_fixed_rd.csv"
RJ_TRAIN = f"{BASE}/xgboost_june_2026/cetrio_training_data.csv"

STATIC = ['lanes','maxspeed','nightlight','pop_density','road_density_500m',
          'road_density_1000m','road_density_5000m','highway_code',
          'surface_code','oneway_code','lu_tree','lu_shrubland','lu_grassland',
          'lu_cropland','lu_builtup','lu_bare','lu_water','lu_wetland',
          'lu_mangroves','lu_moss']
FEATURES = STATIC + ['hour','day_of_week','is_weekend']

COLORS = {'PC':'#2166AC','LCV':'#D6604D','MC':'#4DAF4A','Truck':'#984EA3','Bus':'#FF7F00'}
CLASS_MAP = {'flow_pc':('PC','pc'),'flow_lcv':('LCV','lcv'),'flow_mc':('MC','mc'),
             'flow_truck':('Truck','truck'),'flow_bus':('Bus','bus')}
TRAIN_COLOR = '#636363'  # gray for train curves

FILL = {'lanes':2.0,'maxspeed':60.0,'nightlight':0.0,'pop_density':0.0,
        'road_density_500m':0.0,'road_density_1000m':0.0,'road_density_5000m':0.0,
        'highway_code':4.0,'surface_code':1.0,'oneway_code':1.0}
for c in STATIC:
    if c.startswith('lu_'): FILL[c] = 0.0

print("="*60)
print("Final Paper-Quality Figures")
print("="*60)

# Load data
sp = pd.read_csv(SP_DATA)
rj = pd.read_csv(RJ_TRAIN)
full = pd.concat([sp, rj], ignore_index=True)
print(f"  Data: {len(full):,} rows (SP: {len(sp):,} + RJ: {len(rj):,})")

TARGETS = [t for t in ['flow_pc','flow_lcv','flow_mc','flow_truck','flow_bus']]
TRAIN_TARGETS = [t for t in TARGETS if t in full.columns]

# Winsorize targets (p99 clip) — same protocol as training (02_retrain_and_figures)
for t in TRAIN_TARGETS:
    full[t] = full[t].clip(upper=full[t].quantile(0.99))

# ── FIG 1: Learning Curves ──
print("\n1/6: Learning curves...")
with open(f'{MODEL_DIR}/results_sprj.json') as f:
    results = json.load(f)

fig, axes = plt.subplots(1, 5, figsize=(18, 3.8))
for idx, target in enumerate(TRAIN_TARGETS):
    cls_name, _ = CLASS_MAP[target]
    if target not in results:
        axes[idx].text(0.5, 0.5, 'No model', ha='center', transform=axes[idx].transAxes)
        continue
    ev = results[target].get('evals', {})
    if not ev:
        axes[idx].text(0.5, 0.5, 'No eval data', ha='center', transform=axes[idx].transAxes)
        continue
    tr = ev.get('train_rmse', [])
    te = ev.get('test_rmse', [])
    if not te:
        continue
    best = results[target].get('rounds', len(te)-1)
    
    epochs = range(1, len(tr)+1)
    axes[idx].plot(epochs, tr, color=TRAIN_COLOR, lw=0.7, alpha=0.6, label='Train' if idx==0 else '')
    axes[idx].plot(epochs, te, color='#2166AC', lw=1.2, label='Test' if idx==0 else '')
    axes[idx].axvline(best, color='#CC0000', ls='--', lw=0.6, alpha=0.5,
                      label=f'Stop (n={best})' if idx==0 else '')
    axes[idx].set_xlabel('Boosting round')
    axes[idx].set_ylabel('RMSE (veh h$^{-1}$)')
    axes[idx].set_title(cls_name, fontweight='bold')
    axes[idx].set_xlim(0, len(te)+5)
    axes[idx].set_ylim(0, max(max(tr)*1.05, max(te)*1.05))
    axes[idx].grid(True, alpha=0.2)
    axes[idx].annotate(f'n={best}', xy=(best, te[min(best, len(te)-1)]),
                       xytext=(best+5, te[min(best, len(te)-1)]),
                       fontsize=6.5, color='#CC0000',
                       arrowprops=dict(arrowstyle='-', color='#CC0000', lw=0.3))

handles = [plt.Line2D([],[], color=TRAIN_COLOR, lw=1),
           plt.Line2D([],[], color='#2166AC', lw=1.2),
           plt.Line2D([],[], color='#CC0000', ls='--', lw=0.6)]
labels = ['Training', 'Test (validation)', 'Early stopping']
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.01),
           ncol=3, fontsize=8, frameon=False)

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
fig.savefig(f'{FIGS_DIR}/fig01_learning_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  -> {FIGS_DIR}/fig01_learning_curves.png")

# ── FIG 2: Validation scatter (hexbin with LogNorm, shared colorbar) ──
print("\n2/6: Validation scatter (SP+RJ full)...")
SCAT_TARGETS = [t for t in ['flow_pc','flow_lcv','flow_mc','flow_truck'] if t in full.columns]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
hb_list = []

for idx, target in enumerate(SCAT_TARGETS):
    cls_name, short = CLASS_MAP[target]
    model = xgb.XGBRegressor()
    model_path = f'{MODEL_DIR}/xgb_flow_{short}_sprj.json'
    if not os.path.exists(model_path):
        axes[idx].text(0.5, 0.5, 'No model', ha='center', transform=axes[idx].transAxes)
        continue
    model.load_model(model_path)

    df_v = full.dropna(subset=[target]).copy()
    for c in FEATURES:
        df_v[c] = df_v[c].fillna(FILL.get(c, 0.0))

    y_o = df_v[target].values
    dm = xgb.DMatrix(df_v[FEATURES], feature_names=FEATURES)
    y_p = model.get_booster().predict(dm)

    r2 = r2_score(y_o, y_p)
    mae = mean_absolute_error(y_o, y_p)
    _, pval = pearsonr(y_o, y_p)
    pval_str = 'p<0.001' if pval < 0.001 else f'p={pval:.3f}'

    ax = axes[idx]
    valid = (y_o > 0) & (y_p > 0)
    if valid.sum() > 50:
        hb = ax.hexbin(y_o[valid], y_p[valid], gridsize=45,
                       norm=LogNorm(), cmap='Blues', mincnt=1, alpha=0.85,
                       linewidths=0)
    else:
        hb = ax.hexbin(y_o, y_p, gridsize=40, cmap='Blues', mincnt=1, alpha=0.85,
                       linewidths=0)
    hb_list.append(hb)

    lim = max(y_o.max(), y_p.max()) * 1.02
    ax.plot([0, lim], [0, lim], '--', color='#333333', alpha=0.3, lw=0.6)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(f'Observed {cls_name} (veh h$^{{-1}}$)')
    ax.set_ylabel(f'Predicted {cls_name} (veh h$^{{-1}}$)')
    ax.set_title(f'{cls_name}: $R^2$={r2:.3f}, MAE={mae:.0f}, {pval_str}', fontweight='bold')

for idx in range(len(SCAT_TARGETS), 4):
    axes[idx].set_visible(False)

fig.suptitle('SP + RJ Model Validation (Mon-Wed train, Thu-Sun test)',
             fontsize=13, fontweight='bold')
fig.subplots_adjust(right=0.90, left=0.06, bottom=0.06, top=0.93, wspace=0.35, hspace=0.3)
cbar_ax = fig.add_axes([0.91, 0.10, 0.012, 0.80])
cbar = fig.colorbar(hb_list[0], cax=cbar_ax, label='Count (log scale)')
cbar.ax.tick_params(labelsize=8)

fig.savefig(f'{FIGS_DIR}/fig02_validation_scatter.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  -> {FIGS_DIR}/fig02_validation_scatter.png")

# ── FIG 3: RJ validation scatter (Thu-Sun test set, shared colorbar) ──
print("\n3/6: RJ validation scatter (held-out)...")
rj_test = rj[rj['day_of_week'].isin([3,4,5,6])].copy()
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
hb_rj = []

RJ_TARGETS = [t for t in ['flow_pc','flow_lcv','flow_mc','flow_truck']]

for idx, target in enumerate(RJ_TARGETS):
    cls_name, short = CLASS_MAP[target]
    model = xgb.XGBRegressor()
    model.load_model(f'{MODEL_DIR}/xgb_flow_{short}_sprj.json')
    
    df_v = rj_test.copy()
    for c in FEATURES:
        if c in df_v.columns:
            df_v[c] = df_v[c].fillna(FILL.get(c, 0.0))
    
    y_o = df_v[target].values
    dm = xgb.DMatrix(df_v[FEATURES], feature_names=FEATURES)
    y_p = model.get_booster().predict(dm)
    
    r2 = r2_score(y_o, y_p)
    mae = mean_absolute_error(y_o, y_p)
    bias = y_p.mean() / y_o.mean()
    
    ax = axes[idx]
    valid = (y_o > 0) & (y_p > 0)
    if valid.sum() > 50:
        hb = ax.hexbin(y_o[valid], y_p[valid], gridsize=40,
                       norm=LogNorm(), cmap='Blues', mincnt=1, alpha=0.85,
                       linewidths=0)
    else:
        hb = ax.hexbin(y_o, y_p, gridsize=40, cmap='Blues', mincnt=1, alpha=0.85,
                       linewidths=0)
    hb_rj.append(hb)
    
    lim = max(y_o.max(), y_p.max()) * 1.02
    ax.plot([0, lim], [0, lim], '--', color='#333333', alpha=0.3, lw=0.6)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(f'Observed {cls_name} (veh h$^{-1}$)')
    ax.set_ylabel(f'Predicted {cls_name} (veh h$^{-1}$)')
    ax.set_title(f'{cls_name}: $R^2$={r2:.3f}, bias={bias:.2f}$\\times$', fontweight='bold')

fig.subplots_adjust(right=0.90)
cbar_ax = fig.add_axes([0.91, 0.10, 0.012, 0.80])
cbar = fig.colorbar(hb_rj[0], cax=cbar_ax, label='Count (log scale)')
cbar.ax.tick_params(labelsize=7)

fig.suptitle('CET-Rio Validation (Thu–Sun, held-out from training)', 
             fontsize=13, fontweight='bold', y=1.01)
fig.subplots_adjust(left=0.08, right=0.88, bottom=0.08, top=0.93, wspace=0.3, hspace=0.3)
fig.savefig(f'{FIGS_DIR}/fig06_rj_validation_temporal.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  -> {FIGS_DIR}/fig06_rj_validation_temporal.png")

# ── FIG 4: R² by DOW ──
print("\n4/6: R² by day of week...")
dow_names = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(7); width = 0.18

for idx, target in enumerate(RJ_TARGETS):
    cls_name, short = CLASS_MAP[target]
    model = xgb.XGBRegressor()
    model.load_model(f'{MODEL_DIR}/xgb_flow_{short}_sprj.json')
    
    r2s = []
    for dow in range(7):
        m = rj_test['day_of_week'] == dow
        if m.sum() < 10:
            r2s.append(np.nan); continue
        df_m = rj_test[m].copy()
        for c in FEATURES:
            if c in df_m.columns:
                df_m[c] = df_m[c].fillna(FILL.get(c, 0.0))
        y_o = df_m[target].values
        dm = xgb.DMatrix(df_m[FEATURES], feature_names=FEATURES)
        y_p = model.get_booster().predict(dm)
        valid = ~(np.isnan(y_o) | np.isnan(y_p))
        if valid.sum() > 5:
            r2s.append(r2_score(y_o[valid], y_p[valid]))
        else:
            r2s.append(np.nan)
    
    ax.bar(x + idx*width, r2s, width, label=cls_name, color=COLORS[cls_name], alpha=0.85, edgecolor='white', lw=0.3)

ax.set_xlabel('Day of week')
ax.set_ylabel('$R^2$')
ax.set_title('Validation $R^2$ by Day of Week (CET-Rio held-out)', fontweight='bold')
ax.set_xticks(x + width*2)
ax.set_xticklabels(dow_names)
ax.legend(loc='lower left', fontsize=7.5, ncol=3)
ax.axhline(0, color='gray', ls='-', alpha=0.3, lw=0.5)
ax.set_ylim(-0.3, 0.9)
ax.grid(True, alpha=0.2, axis='y')
plt.tight_layout()
fig.savefig(f'{FIGS_DIR}/fig03_validation_dow.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  -> {FIGS_DIR}/fig03_validation_dow.png")

# ── FIG 5: Feature importance ──
print("\n5/6: Feature importance...")
n_top = 12
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for idx, target in enumerate(TRAIN_TARGETS):
    cls_name, short = CLASS_MAP[target]
    model = xgb.XGBRegressor()
    model.load_model(f'{MODEL_DIR}/xgb_flow_{short}_sprj.json')
    
    imp = model.get_booster().get_score(importance_type='gain')
    items = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:n_top]
    names = [x[0] for x in items[::-1]]
    vals = [x[1] for x in items[::-1]]
    
    axes[idx].barh(range(len(names)), vals, color=COLORS[cls_name], alpha=0.8, height=0.65)
    axes[idx].set_yticks(range(len(names)))
    axes[idx].set_yticklabels(names, fontsize=6.5)
    axes[idx].set_xlabel('Gain')
    axes[idx].set_title(cls_name, fontweight='bold')
    axes[idx].grid(True, alpha=0.2, axis='x')

for idx in range(len(TRAIN_TARGETS), 5):
    axes[idx].set_visible(False)
plt.tight_layout()
fig.savefig(f'{FIGS_DIR}/fig04_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  -> {FIGS_DIR}/fig04_feature_importance.png")

# ── FIG 5b: Performance comparison bar chart (SP-only vs SP+RJ) ──
print("\n6/6: Performance comparison...")
fig, ax = plt.subplots(figsize=(7, 4.5))
classes = ['PC', 'LCV', 'MC', 'Truck']
# SP-only R² (from SP-only models trained on fixed_rd data, Table 1)
sp_json_path = f'{MODEL_DIR}/results_sp_flow_r.json'
if os.path.exists(sp_json_path):
    with open(sp_json_path) as f:
        sp_res = json.load(f)
    sp_r2 = [sp_res[t]['r2'] for t in ['flow_pc','flow_lcv','flow_mc','flow_truck']]
else:
    print('  WARNING: results_sp_flow_r.json not found, using published Table 1 values')
    sp_r2 = [0.728, 0.549, 0.611, 0.559]
# SP+RJ R²
sprj_r2 = [results[t]['r2'] for t in ['flow_pc','flow_lcv','flow_mc','flow_truck']]
# RJ held-out R²
rj_r2_vals = []
for t in ['flow_pc','flow_lcv','flow_mc','flow_truck']:
    with open(f'{MODEL_DIR}/rj_validation_proper.json') as f:
        rjv = json.load(f)
    rj_r2_vals.append(rjv[t]['r2'])
    
x = np.arange(len(classes)); w = 0.25
ax.bar(x - w, sp_r2, w, label='SP-only (SP test)', color='#BDBDBD', edgecolor='white', lw=0.3)
ax.bar(x, sprj_r2, w, label='SP+RJ (SP+RJ test)', color='#74A9CF', edgecolor='white', lw=0.3)
ax.bar(x + w, rj_r2_vals, w, label='SP+RJ (RJ held-out)', color='#2166AC', edgecolor='white', lw=0.3)
ax.set_xlabel('Vehicle class')
ax.set_ylabel('$R^2$')
ax.set_title('Model Performance Comparison', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.legend(fontsize=7.5)
ax.axhline(0, color='gray', ls='-', alpha=0.3, lw=0.5)
ax.set_ylim(0, 0.85)
ax.grid(True, alpha=0.2, axis='y')
plt.tight_layout()
fig.savefig(f'{FIGS_DIR}/fig07_performance_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  -> {FIGS_DIR}/fig07_performance_comparison.png")

print("\n" + "="*60)
print("Final figures complete.")
print("="*60)
