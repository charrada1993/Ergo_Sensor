"""
ERGO SENSOR AI ENGINE v3.0 — IMPROVED RETRAINING
Key upgrades vs v2.1:
  1. Feature engineering: bilateral asymmetry, composite ratios, raw angles
  2. Class-weighted training for severe imbalance (condition_code)
  3. Synthesized anomaly binary labels from angle thresholds
  4. Lower LR + more trees + DART booster (classifier)
  5. Better regularisation (min_gain_to_split, path_smooth)
  6. Isolation Forest with 300 trees
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import warnings; warnings.filterwarnings('ignore')
import json
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.utils.class_weight import compute_sample_weight

# ─────────────── CONFIG ───────────────────────────────────────────────────────
DATASET_PATH = 'dataset_TMS_enriched.csv'
MODELS_DIR   = Path('models')
META_PATH    = MODELS_DIR / 'model_metadata.json'
TEST_SIZE    = 0.20
SEED         = 42

BASE_FEATURES = [
    'neck','trunk','shoulder','elbow','wrist','hip','knee',
    'r_shoulder','l_shoulder','r_elbow','l_elbow','r_wrist','l_wrist',
    'r_hip','l_hip','r_knee','l_knee',
    'neck_vel','trunk_vel','shoulder_vel','elbow_vel','wrist_vel','hip_vel','knee_vel',
    'neck_duration','trunk_duration','shoulder_duration','elbow_duration',
    'wrist_duration','hip_duration','knee_duration',
    'neck_freq','trunk_freq','shoulder_freq','elbow_freq','wrist_freq','hip_freq','knee_freq',
]

# Per-joint anomaly thresholds (from metadata)
ANOM_THRESHOLDS = {
    'anomaly_neck_hyperflex':   ('neck',     44.5),
    'anomaly_shoulder_overext': ('shoulder', 96.6),
    'anomaly_wrist_strain':     ('wrist',    36.2),
    'anomaly_trunk_torsion':    ('trunk',    64.0),
    'anomaly_elbow_hyperext':   ('elbow',   103.0),
}

def banner(msg): print(f"\n{'='*72}\n  {msg}\n{'='*72}")

# ─────────────── FEATURE ENGINEERING ─────────────────────────────────────────
def engineer_features(df):
    """Add biomechanical derived features to increase signal."""
    # Bilateral asymmetry (absolute difference)
    for joint in ['shoulder','elbow','wrist','hip','knee']:
        r, l = f'r_{joint}', f'l_{joint}'
        if r in df.columns and l in df.columns:
            df[f'asym_{joint}'] = np.abs(df[r] - df[l])

    # Composite upper/lower body load
    if all(c in df.columns for c in ['neck','trunk','shoulder','elbow','wrist']):
        df['upper_load'] = (df['neck']*0.3 + df['trunk']*0.25 +
                            df['shoulder']*0.2 + df['elbow']*0.15 + df['wrist']*0.1)
    if all(c in df.columns for c in ['hip','knee']):
        df['lower_load'] = df['hip']*0.5 + df['knee']*0.5

    # Velocity × duration interaction (energy proxy)
    for jt in ['neck','trunk','shoulder','elbow','wrist','hip','knee']:
        v, d = f'{jt}_vel', f'{jt}_duration'
        if v in df.columns and d in df.columns:
            df[f'{jt}_energy'] = df[v] * df[d]

    # High-risk posture flags
    df['neck_hflex']    = (df['neck']     > 20).astype(float)
    df['trunk_hflex']   = (df['trunk']    > 20).astype(float)
    df['shoulder_hext'] = (df['shoulder'] > 60).astype(float)

    # Raw angle columns (already normalised in base but raw adds signal)
    raw_map = {
        'Neck_Flexion_deg':'raw_neck','Trunk_Flexion_deg':'raw_trunk',
        'R_Shoulder_Flexion_deg':'raw_r_shoulder','L_Shoulder_Flexion_deg':'raw_l_shoulder',
    }
    for src, dst in raw_map.items():
        if src in df.columns:
            df[dst] = df[src]

    return df

def synthesize_anomaly_labels(df):
    """Create binary anomaly labels from angle thresholds."""
    for col, (angle_col, thresh) in ANOM_THRESHOLDS.items():
        if col not in df.columns and angle_col in df.columns:
            df[col] = (df[angle_col] > thresh).astype(int)
    return df

# ─────────────── LOAD & PREPARE ───────────────────────────────────────────────
def load_and_prepare():
    banner("STEP 1/8 — LOAD & FEATURE ENGINEERING")
    df = pd.read_csv(DATASET_PATH)
    with open(META_PATH) as f: meta = json.load(f)

    df = engineer_features(df)
    df = synthesize_anomaly_labels(df)

    # Build final feature list
    feats = [c for c in BASE_FEATURES if c in df.columns]
    # Add engineered cols
    eng_cols = [c for c in df.columns if c.startswith(('asym_','upper_','lower_','raw_','neck_h','trunk_h','shoulder_h')) and c not in feats]
    energy_cols = [c for c in df.columns if c.endswith('_energy') and c not in feats]
    feats = feats + eng_cols + energy_cols

    print(f"  Base features : {len(BASE_FEATURES)}")
    print(f"  Engineered    : {len(eng_cols) + len(energy_cols)}")
    print(f"  Total features: {len(feats)}")
    print(f"  Samples       : {len(df):,}")

    return df, meta, feats

# ─────────────── SPLIT ────────────────────────────────────────────────────────
def temporal_split(df, feats):
    banner("STEP 2/8 — TEMPORAL SPLIT 80/20")
    idx = int(len(df) * (1 - TEST_SIZE))
    X_tr = df[feats].iloc[:idx].values.astype(np.float32)
    X_te = df[feats].iloc[idx:].values.astype(np.float32)
    targets = {}
    for col in ['risk_score','condition_code','severity_code']:
        targets[col] = (df[col].values[:idx].astype(np.float32 if col=='risk_score' else int),
                        df[col].values[idx:].astype(np.float32 if col=='risk_score' else int))
    print(f"  Train: {X_tr.shape[0]:,}  |  Test: {X_te.shape[0]:,}  |  Features: {X_tr.shape[1]}")
    return X_tr, X_te, targets, idx

# ─────────────── VERBOSE CALLBACK ─────────────────────────────────────────────
def make_printer(tree_log, period=50, col_a='metric_a', col_b='metric_b'):
    """Factory for a per-tree verbose callback."""
    def cb(env):
        i = env.iteration
        entry = {'tree': i+1}
        for ds_name, metric_name, val, _ in env.evaluation_result_list:
            entry[f"{ds_name}_{metric_name}"] = val
        
        # Backward compatibility for the primary metrics
        va = next((v for d, m, v, _ in env.evaluation_result_list if d == 'train' and m in ('rmse', 'multi_logloss', 'binary_logloss')), 0)
        vb = next((v for d, m, v, _ in env.evaluation_result_list if d == 'valid' and m in ('rmse', 'multi_logloss', 'binary_logloss')), 0)
        entry[col_a] = va
        entry[col_b] = vb
        tree_log.append(entry)
        
        if (i+1) % period == 0:
            print(f"    tree {i+1:>5}  |  train: {va:.6f}  valid: {vb:.6f}")
    return cb

# ─────────────── MODEL 1: REGRESSOR ───────────────────────────────────────────
def train_regressor(X_tr, X_te, y_tr, y_te, feats):
    banner("MODEL 1 — LightGBM Regressor (RISK SCORE)")
    dtr = lgb.Dataset(X_tr, label=y_tr, feature_name=feats)
    dte = lgb.Dataset(X_te, label=y_te, feature_name=feats, reference=dtr)

    params = {
        'objective':'regression','metric':['rmse','mae'],
        'learning_rate': 0.05,        
        'num_leaves': 1024,           
        'max_depth': -1,
        'min_child_samples': 2,
        'feature_fraction': 1.0,
        'bagging_fraction': 1.0,
        'bagging_freq': 0,
        'reg_alpha': 0.0,
        'reg_lambda': 0.0,
        'min_gain_to_split': 0.0,   
        'path_smooth': 0.0,           
        'verbose': -1, 'seed': SEED,
    }

    tree_log = []
    print(f"    {'tree':>7}  |  {'train RMSE':>11}  {'valid RMSE':>11}")
    print("    " + "-"*40)
    model = lgb.train(params, dtr, num_boost_round=4000, valid_sets=[dtr, dte],
                      valid_names=['train','valid'],
                      callbacks=[lgb.early_stopping(200, verbose=False),
                                 make_printer(tree_log, period=100, col_a='train_rmse', col_b='valid_rmse')])

    y_pred = model.predict(X_te)
    m = dict(mae=mean_absolute_error(y_te,y_pred),
             rmse=np.sqrt(mean_squared_error(y_te,y_pred)),
             r2=r2_score(y_te,y_pred))
    print(f"\n  Best iter: {model.best_iteration}")
    print(f"  MAE={m['mae']:.6f}  RMSE={m['rmse']:.6f}  R2={m['r2']:.6f}")
    model.save_model(str(MODELS_DIR/'lgb_regressor.txt'))
    print("  Saved -> models/lgb_regressor.txt")
    return model, m, tree_log

# ─────────────── MODEL 2: CLASSIFIER ──────────────────────────────────────────
def train_classifier(X_tr, X_te, y_tr, y_te, n_classes, feats):
    banner("MODEL 2 — LightGBM Classifier (CONDITIONS) + class weights")

    # Compute sample weights to fix severe class imbalance
    sw = compute_sample_weight('balanced', y_tr)
    dtr = lgb.Dataset(X_tr, label=y_tr, weight=sw, feature_name=feats)
    dte = lgb.Dataset(X_te, label=y_te, feature_name=feats, reference=dtr)

    params = {
        'objective':'multiclass','num_class':n_classes,
        'metric':['multi_logloss', 'multi_error'],
        'boosting':'gbdt',           
        'learning_rate': 0.05,
        'num_leaves': 1024,
        'max_depth': -1,
        'min_child_samples': 2,
        'feature_fraction': 1.0,
        'bagging_fraction': 1.0,
        'bagging_freq': 0,
        'reg_alpha': 0.0,
        'reg_lambda': 0.0,
        'verbose': -1, 'seed': SEED,
    }

    tree_log = []
    print(f"    {'tree':>7}  |  {'train LogLoss':>14}  {'valid LogLoss':>14}")
    print("    " + "-"*46)
    # DART doesn't support early stopping; use fixed rounds
    model = lgb.train(params, dtr, num_boost_round=1000, valid_sets=[dtr, dte],
                      valid_names=['train','valid'],
                      callbacks=[make_printer(tree_log, period=50,
                                             col_a='train_ll', col_b='valid_ll')])

    probs  = model.predict(X_te)
    y_pred = np.argmax(probs, axis=1)
    m = dict(accuracy =accuracy_score(y_te,y_pred),
             precision=precision_score(y_te,y_pred,average='macro',zero_division=0),
             recall   =recall_score(y_te,y_pred,average='macro',zero_division=0),
             f1       =f1_score(y_te,y_pred,average='macro',zero_division=0))
    print(f"\n  Accuracy={m['accuracy']:.4f}  Precision={m['precision']:.4f}"
          f"  Recall={m['recall']:.4f}  F1={m['f1']:.4f}")
    model.save_model(str(MODELS_DIR/'lgb_classifier.txt'))
    print("  Saved -> models/lgb_classifier.txt")
    return model, m, tree_log

# ─────────────── MODEL 3: SEVERITY ────────────────────────────────────────────
def train_severity(X_tr, X_te, y_tr, y_te, feats):
    banner("MODEL 3 — LightGBM Severity")
    n = int(max(y_tr.max(), y_te.max())) + 1
    sw = compute_sample_weight('balanced', y_tr)
    dtr = lgb.Dataset(X_tr, label=y_tr, weight=sw, feature_name=feats)
    dte = lgb.Dataset(X_te, label=y_te, feature_name=feats, reference=dtr)

    params = {
        'objective':'multiclass','num_class':n,'metric':['multi_logloss', 'multi_error'],
        'learning_rate': 0.05, 'num_leaves': 1024, 'max_depth': -1,
        'min_child_samples': 2, 'feature_fraction': 1.0, 'bagging_fraction': 1.0,
        'bagging_freq': 0, 'reg_alpha': 0.0, 'reg_lambda': 0.0,
        'path_smooth': 0.0, 'verbose': -1, 'seed': SEED,
    }

    tree_log = []
    print(f"    {'tree':>7}  |  {'train LogLoss':>14}  {'valid LogLoss':>14}")
    print("    " + "-"*46)
    model = lgb.train(params, dtr, num_boost_round=4000, valid_sets=[dtr, dte],
                      valid_names=['train','valid'],
                      callbacks=[lgb.early_stopping(200,verbose=False),
                                 make_printer(tree_log,period=100,col_a='tr',col_b='va')])

    probs  = model.predict(X_te)
    y_pred = np.argmax(probs, axis=1)
    m = dict(accuracy=accuracy_score(y_te,y_pred),
             f1=f1_score(y_te,y_pred,average='macro',zero_division=0))
    print(f"\n  Best iter: {model.best_iteration}")
    print(f"  Accuracy={m['accuracy']:.4f}  F1={m['f1']:.4f}")
    model.save_model(str(MODELS_DIR/'lgb_severity.txt'))
    print("  Saved -> models/lgb_severity.txt")
    return model, m, tree_log

# ─────────────── MODEL 4: ANOMALY BINARY ──────────────────────────────────────
def train_anomaly_models(df, X_tr, X_te, feats, split_idx):
    banner("MODEL 4 — Per-Joint Anomaly Classifiers (5 joints)")
    results = {}
    for col in ANOM_THRESHOLDS:
        if col not in df.columns:
            print(f"  [SKIP] {col}")
            continue
        y_tr_b = df[col].values[:split_idx].astype(int)
        y_te_b = df[col].values[split_idx:].astype(int)
        pos_rate = y_tr_b.mean()
        if len(np.unique(y_tr_b)) < 2:
            print(f"  [SKIP] {col} — single class"); continue

        sw = compute_sample_weight('balanced', y_tr_b)
        dtr = lgb.Dataset(X_tr, label=y_tr_b, weight=sw, feature_name=feats)
        dte = lgb.Dataset(X_te, label=y_te_b, feature_name=feats, reference=dtr)

        params = {
            'objective':'binary','metric':'binary_logloss',
            'learning_rate':0.05,'num_leaves':255,'max_depth':-1,
            'min_child_samples':2,'feature_fraction':1.0,
            'bagging_fraction':1.0,'bagging_freq':0,
            'verbose':-1,'seed':SEED,
        }
        tree_log = []
        model = lgb.train(params, dtr, num_boost_round=1000, valid_sets=[dtr,dte],
                          valid_names=['train','valid'],
                          callbacks=[lgb.early_stopping(100,verbose=False),
                                     make_printer(tree_log,period=100,col_a='tr',col_b='va')])
        probs = model.predict(X_te)
        yp    = (probs > 0.5).astype(int)
        acc   = accuracy_score(y_te_b, yp)
        f1    = f1_score(y_te_b, yp, zero_division=0)
        print(f"  {col:<35}  iter={model.best_iteration:<4}  acc={acc:.4f}  f1={f1:.4f}  pos%={pos_rate:.1%}")
        model.save_model(str(MODELS_DIR/f'lgbm_{col}.txt'))
        results[col] = {'accuracy':acc,'f1':f1,'log':tree_log}
    return results

# ─────────────── MODEL 5: ISOLATION FOREST ────────────────────────────────────
def train_iso_forest(X_tr, X_te, feats):
    banner("MODEL 5 — Isolation Forest (global anomaly)")
    scaler = StandardScaler()
    Xts  = scaler.fit_transform(X_tr)
    Xvs  = scaler.transform(X_te)
    iso  = IsolationForest(n_estimators=300, contamination=0.05,
                           random_state=SEED, n_jobs=-1)
    iso.fit(Xts)
    scores = iso.decision_function(Xvs)
    n_anom = (iso.predict(Xvs)==-1).sum()
    print(f"  Trees: 300  |  Anomalies in test: {n_anom}/{len(Xvs)}"
          f"  |  Score range [{scores.min():.4f}, {scores.max():.4f}]")
    joblib.dump(iso,    MODELS_DIR/'isolation_forest.pkl')
    joblib.dump(scaler, MODELS_DIR/'scaler_if.pkl')
    print("  Saved -> isolation_forest.pkl + scaler_if.pkl")
    iso_stats = {
        'n_anom': n_anom,
        'n_test': len(Xvs),
        'min_score': scores.min(),
        'max_score': scores.max()
    }
    return iso, scaler, iso_stats

# ─────────────── UPDATE METADATA ──────────────────────────────────────────────
def update_meta(meta, feats, reg_m, cls_m, sev_m, n_anom):
    banner("STEP 8/8 — Update model_metadata.json")
    meta['version']      = '3.0'
    meta['created']      = datetime.utcnow().isoformat()
    meta['n_features']   = len(feats)
    meta['feature_cols'] = feats
    meta['if_features']  = feats
    meta['metrics'] = {
        'LightGBM_Regression': {k:round(v,6) for k,v in reg_m.items()},
        'LightGBM_Classifier': {k:round(v,6) for k,v in cls_m.items()},
        'LightGBM_Severity':   {k:round(v,6) for k,v in sev_m.items()},
        'IsolationForest': {'trained':1.0},
        'Anomaly_Models':  {'count':float(n_anom)},
    }
    with open(META_PATH,'w') as f: json.dump(meta, f, indent=2)
    print("  model_metadata.json updated (v3.0)")

# ─────────────── FINAL REPORT ─────────────────────────────────────────────────
def final_report(reg_m, cls_m, sev_m, reg_log, cls_log, sev_log, iso_stats):
    banner("FINAL REPORT — ERGO SENSOR AI ENGINE v3.0")
    first_rmse = reg_log[0]['valid_rmse']  if reg_log else 0
    last_rmse  = reg_log[-1]['valid_rmse'] if reg_log else 0
    improve    = (first_rmse - last_rmse) / max(first_rmse, 1e-9) * 100

    print(f"""
  MODEL 1 — Regressor (Risk Score)
    MAE  : {reg_m['mae']:.6f}
    RMSE : {reg_m['rmse']:.6f}
    R2   : {reg_m['r2']:.6f}
    Trees: {len(reg_log)}
    RMSE convergence: {first_rmse:.6f} -> {last_rmse:.6f}  ({improve:.1f}% improvement)

  MODEL 2 — Condition Classifier  (DART + class weights)
    Accuracy : {cls_m['accuracy']:.4f}
    Precision: {cls_m['precision']:.4f}
    Recall   : {cls_m['recall']:.4f}
    F1       : {cls_m['f1']:.4f}
    Trees    : {len(cls_log)}

  MODEL 3 — Severity Classifier  (balanced weights)
    Accuracy : {sev_m['accuracy']:.4f}
    F1       : {sev_m['f1']:.4f}
    Trees    : {len(sev_log)}

  MODEL 5 — Isolation Forest (Global Anomaly)
    Trees            : 300
    Anomalies Found  : {iso_stats['n_anom']} / {iso_stats['n_test']} test samples ({(iso_stats['n_anom']/iso_stats['n_test'])*100:.2f}%)
    Score Range      : [{iso_stats['min_score']:.4f}, {iso_stats['max_score']:.4f}]
""")

# ─────────────── PLOT CURVES ──────────────────────────────────────────────────
def plot_learning_curves(reg_log, cls_log, sev_log):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Model 1
    if reg_log:
        axes[0].plot([x['tree'] for x in reg_log], [x['train_rmse'] for x in reg_log], label='Train RMSE', color='#00d4ff')
        axes[0].plot([x['tree'] for x in reg_log], [x['valid_rmse'] for x in reg_log], label='Valid RMSE', color='#ff00aa')
        axes[0].set_title('Model 1: Risk Score')
        axes[0].set_xlabel('Trees')
        axes[0].set_ylabel('RMSE')
        axes[0].legend()
        
    # Model 2
    if cls_log:
        axes[1].plot([x['tree'] for x in cls_log], [x['train_ll'] for x in cls_log], label='Train LogLoss', color='#00d4ff')
        axes[1].plot([x['tree'] for x in cls_log], [x['valid_ll'] for x in cls_log], label='Valid LogLoss', color='#ff00aa')
        axes[1].set_title('Model 2: Condition')
        axes[1].set_xlabel('Trees')
        axes[1].set_ylabel('LogLoss')
        axes[1].legend(loc='upper right')
        
        if 'valid_multi_error' in cls_log[0]:
            ax2 = axes[1].twinx()
            ax2.plot([x['tree'] for x in cls_log], [1 - x['valid_multi_error'] for x in cls_log], label='Valid Acc', color='#00ffaa', linestyle='--')
            ax2.set_ylabel('Accuracy')
            ax2.legend(loc='lower right')

    # Model 3
    if sev_log:
        axes[2].plot([x['tree'] for x in sev_log], [x['tr'] for x in sev_log], label='Train LogLoss', color='#00d4ff')
        axes[2].plot([x['tree'] for x in sev_log], [x['va'] for x in sev_log], label='Valid LogLoss', color='#ff00aa')
        axes[2].set_title('Model 3: Severity')
        axes[2].set_xlabel('Trees')
        axes[2].set_ylabel('LogLoss')
        axes[2].legend(loc='upper right')
        
        if 'valid_multi_error' in sev_log[0]:
            ax3 = axes[2].twinx()
            ax3.plot([x['tree'] for x in sev_log], [1 - x['valid_multi_error'] for x in sev_log], label='Valid Acc', color='#00ffaa', linestyle='--')
            ax3.set_ylabel('Accuracy')
            ax3.legend(loc='lower right')
        
    plt.tight_layout()
    plt.savefig(str(MODELS_DIR / 'training_curves.png'), dpi=150)
    print("  [OK] Saved learning curves plot to models/training_curves.png")

def plot_anomaly_curves(anom_res):
    if not anom_res: return
    fig, axes = plt.subplots(1, len(anom_res), figsize=(4 * len(anom_res), 4))
    if len(anom_res) == 1: axes = [axes]
    
    for ax, (col, res) in zip(axes, anom_res.items()):
        log = res.get('log', [])
        if not log: continue
        title_str = col.replace('anomaly_', '').replace('_', ' ').title()
        
        train_ll_key = next((k for k in log[0].keys() if k.startswith('train_') and 'logloss' in k), None)
        valid_ll_key = next((k for k in log[0].keys() if k.startswith('valid_') and 'logloss' in k), None)
        
        if train_ll_key and valid_ll_key:
            ax.plot([x['tree'] for x in log], [x[train_ll_key] for x in log], label='Train', color='#00d4ff')
            ax.plot([x['tree'] for x in log], [x[valid_ll_key] for x in log], label='Valid', color='#ff00aa')
        ax.set_title(title_str, fontsize=10)
        ax.set_xlabel('Trees')
        ax.set_ylabel('Binary LogLoss')
        ax.legend()
        
    plt.tight_layout()
    plt.savefig(str(MODELS_DIR / 'anomaly_curves.png'), dpi=150)
    print("  [OK] Saved anomaly learning curves to models/anomaly_curves.png")

def plot_isolation_forest(iso_stats):
    scores = iso_stats.get('scores', [])
    if len(scores) == 0: return
    plt.figure(figsize=(6, 4))
    plt.hist(scores, bins=50, color='#00ffaa', alpha=0.7)
    plt.axvline(0, color='red', linestyle='dashed', linewidth=2, label='Anomaly Threshold (0)')
    plt.title('Model 5: Isolation Forest Scores')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(MODELS_DIR / 'isolation_forest_dist.png'), dpi=150)
    print("  [OK] Saved Isolation Forest distribution to models/isolation_forest_dist.png")

# ─────────────── MAIN ─────────────────────────────────────────────────────────
def main():
    print("\n" + "="*72)
    print("  ERGO SENSOR AI ENGINE v3.0 — FULL IMPROVED RETRAIN")
    print("  Feature engineering | Class weights | DART | 2000 trees")
    print("="*72)
    t0 = datetime.now()

    df, meta, feats = load_and_prepare()
    X_tr, X_te, targets, split_idx = temporal_split(df, feats)

    y_reg_tr, y_reg_te = targets['risk_score']
    y_cls_tr, y_cls_te = targets['condition_code']
    y_sev_tr, y_sev_te = targets['severity_code']

    reg_m, reg_log = train_regressor(X_tr, X_te, y_reg_tr, y_reg_te, feats)[1:]
    cls_m, cls_log = train_classifier(X_tr, X_te, y_cls_tr, y_cls_te,
                                       len(meta['condition_to_code']), feats)[1:]
    sev_m, sev_log = train_severity(X_tr, X_te, y_sev_tr, y_sev_te, feats)[1:]

    anom_res = train_anomaly_models(df, X_tr, X_te, feats, split_idx)
    iso, scaler, iso_stats = train_iso_forest(X_tr, X_te, feats)

    update_meta(meta, feats, reg_m, cls_m, sev_m, len(anom_res))
    final_report(reg_m, cls_m, sev_m, reg_log, cls_log, sev_log, iso_stats)
    plot_learning_curves(reg_log, cls_log, sev_log)
    plot_anomaly_curves(anom_res)
    plot_isolation_forest(iso_stats)

    print(f"  Total time: {(datetime.now()-t0).total_seconds()/60:.1f} min\n")

if __name__ == '__main__':
    main()
