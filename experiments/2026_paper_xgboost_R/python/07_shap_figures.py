#!/usr/bin/env python3
"""
Step 7: SHAP summary figures for all flow-class models and the hourly speed
model (Python mirror of final/R/04_generate_all_shap_figures.R).

Two-panel figure per model: left = mean(|SHAP|) top-15 bar chart (class color),
right = SHAP beeswarm (top 15). Saved to final/pyfigs/ as fig05_shap_*.png.
"""
import os, io, warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

import numpy as np
import pandas as pd
import xgboost as xgb
import shap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ── Paper-quality styling (consistent with 02_retrain_and_figures.py) ──
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
    'flow_pc':   {'label':'PC',    'color':'#2166AC', 'short':'pc',    'shap_title':'Passenger Car Flow'},
    'flow_lcv':  {'label':'LCV',   'color':'#D6604D', 'short':'lcv',   'shap_title':'LCV Flow'},
    'flow_mc':   {'label':'MC',    'color':'#4DAF4A', 'short':'mc',    'shap_title':'Motorcycle Flow'},
    'flow_truck':{'label':'Truck', 'color':'#984EA3', 'short':'truck', 'shap_title':'Truck Flow'},
    'flow_bus':  {'label':'Bus',   'color':'#FF7F00', 'short':'bus',   'shap_title':'Bus Flow'}
}
TARGETS = list(CLASSES.keys())

FILL_VALS = {'lanes':2.0,'maxspeed':60.0,'nightlight':0.0,'pop_density':0.0,
             'road_density_500m':0.0,'road_density_1000m':0.0,'road_density_5000m':0.0,
             'highway_code':4.0,'surface_code':1.0,'oneway_code':1.0}
for c in STATIC:
    if c.startswith('lu_'): FILL_VALS[c] = 0.0


def load_model(path):
    model = xgb.Booster()
    model.load_model(path)
    return model


def compute_shap(model, X_sample):
    explainer = shap.TreeExplainer(model)
    return np.asarray(explainer.shap_values(X_sample))


def make_shap_figure(shap_values, X_sample, color, title, out_path):
    """Two-panel SHAP figure: mean(|SHAP|) bar (left) + beeswarm (right)."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[-15:][::-1]
    top_names = [FEATURES[i] for i in top_idx]
    top_vals = mean_abs[top_idx]

    fig = plt.figure(figsize=(11, 9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.25], wspace=0.3)
    ax_bar = fig.add_subplot(gs[0])
    ax_bee = fig.add_subplot(gs[1])

    ax_bar.barh(range(len(top_vals)), top_vals[::-1], color=color,
                alpha=0.85, height=0.7)
    ax_bar.set_yticks(range(len(top_vals)))
    ax_bar.set_yticklabels(top_names[::-1], fontsize=8)
    ax_bar.set_xlabel('mean(|SHAP value|)')
    ax_bar.set_title('Feature Importance', fontsize=11, fontweight='bold')
    ax_bar.grid(True, alpha=0.3, axis='x')

    # Render the beeswarm on its own figure, then embed into the right panel
    bee_fig = plt.figure(figsize=(6, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=FEATURES,
                      max_display=15, show=False)
    buf = io.BytesIO()
    bee_fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(bee_fig)
    buf.seek(0)
    img = plt.imread(buf)
    ax_bee.imshow(img, aspect='auto')
    ax_bee.axis('off')
    ax_bee.set_title('SHAP Beeswarm', fontsize=11, fontweight='bold')

    fig.suptitle(title, fontsize=13, fontweight='bold')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


print("="*60)
print("Step 7: SHAP Summary Figures (Python)")
print("="*60)

# ── Load SP training data and fill NA features ──
print("\nLoading SP training data...")
if not os.path.exists(SP_DATA):
    print(f"  SP data not found at {SP_DATA} — cannot compute SHAP values. Exiting.")
    raise SystemExit(0)
sp = pd.read_csv(SP_DATA)
print(f"  SP data: {len(sp):,} rows")
for c in FEATURES:
    if c in sp.columns:
        sp[c] = sp[c].fillna(FILL_VALS.get(c, 0.0))

n_sample = min(2000, len(sp))
X_sample = sp.sample(n=n_sample, random_state=42)[FEATURES]
print(f"  SHAP sample: {n_sample:,} rows (random_state=42)")

# ── Flow SHAP figures ──
print("\n=== Generating Flow Model SHAP Figures ===")
for target in TARGETS:
    cls = CLASSES[target]
    model_path = f"{MODEL_DIR}/xgb_flow_{cls['short']}_sprj.json"
    if not os.path.exists(model_path):
        model_path = f"{MODEL_DIR}/xgb_flow_{cls['short']}_sp.json"
    if not os.path.exists(model_path):
        print(f"  SKIP {cls['label']}: model not found")
        continue

    model = load_model(model_path)
    shap_values = compute_shap(model, X_sample)
    out_path = f"{FIGS_DIR}/fig05_shap_{cls['short']}.png"
    make_shap_figure(shap_values, X_sample, cls['color'],
                     f"{cls['shap_title']} — SHAP Summary", out_path)

# ── Speed SHAP figure ──
print("\n=== Generating Speed Model SHAP Figure ===")
speed_model_path = f"{MODEL_DIR}/xgb_speed_hourly.json"
if os.path.exists(speed_model_path):
    speed_model = load_model(speed_model_path)
    shap_values = compute_shap(speed_model, X_sample)
    make_shap_figure(shap_values, X_sample, '#E41A1C',
                     "Hourly Speed Model — SHAP Summary",
                     f"{FIGS_DIR}/fig05_shap_speed.png")
else:
    print(f"  Speed model not found at {speed_model_path}")

print("\nDone. All SHAP figures generated.")
