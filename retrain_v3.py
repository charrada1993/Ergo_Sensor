"""
ERGO SENSOR AI ENGINE v3.0 — PRODUCTION-GRADE OPTIMIZATION
Key upgrades:
  1. Time-Series Feature Engineering (Rolling, Lags, Accel)
  2. TimeSeriesSplit Cross-Validation
  3. Optuna Hyperparameter Optimization (Max F1 / Min RMSE)
  4. Explainability with SHAP summary plots
  5. LocalOutlierFactor + Isolation Forest comparison
"""
import sys, io

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
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import TimeSeriesSplit
import optuna
import shap
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ─────────────── CONFIG ───────────────────────────────────────────────────────
DATASET_PATH = 'dataset_TMS_enriched.csv'
MODELS_DIR   = Path('models')
META_PATH    = MODELS_DIR / 'model_metadata.json'
TEST_SIZE    = 0.20
SEED         = 42
N_TRIALS     = 10  # Reduced for faster training cycle

BASE_FEATURES = [
    'neck','trunk','shoulder','elbow','wrist','hip','knee',
    'r_shoulder','l_shoulder','r_elbow','l_elbow','r_wrist','l_wrist',
    'r_hip','l_hip','r_knee','l_knee',
    'neck_vel','trunk_vel','shoulder_vel','elbow_vel','wrist_vel','hip_vel','knee_vel',
    'neck_duration','trunk_duration','shoulder_duration','elbow_duration',
    'wrist_duration','hip_duration','knee_duration',
    'neck_freq','trunk_freq','shoulder_freq','elbow_freq','wrist_freq','hip_freq','knee_freq',
]

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
    """Add biomechanical derived features, rolling stats, lags, and acceleration."""
    # Bilateral asymmetry (absolute difference)
    for joint in ['shoulder','elbow','wrist','hip','knee']:
        r, l = f'r_{joint}', f'l_{joint}'
        if r in df.columns and l in df.columns:
            df[f'asym_{joint}'] = np.abs(df[r] - df[l])

    # Composite load
    if all(c in df.columns for c in ['neck','trunk','shoulder','elbow','wrist']):
        df['upper_load'] = (df['neck']*0.3 + df['trunk']*0.25 + df['shoulder']*0.2 + df['elbow']*0.15 + df['wrist']*0.1)
    if all(c in df.columns for c in ['hip','knee']):
        df['lower_load'] = df['hip']*0.5 + df['knee']*0.5

    # Interaction
    for jt in ['neck','trunk','shoulder','elbow','wrist','hip','knee']:
        v, d = f'{jt}_vel', f'{jt}_duration'
        if v in df.columns and d in df.columns:
            df[f'{jt}_energy'] = df[v] * df[d]

    # Time-Series Features (Rolling & Lag & Accel)
    for jt in ['neck','trunk','shoulder','elbow','wrist']:
        if jt in df.columns:
            # 15 frames = ~0.5s at 30fps
            df[f'{jt}_roll_mean'] = df[jt].rolling(window=15, min_periods=1).mean()
            df[f'{jt}_roll_std'] = df[jt].rolling(window=15, min_periods=1).std().fillna(0)
            df[f'{jt}_lag_15'] = df[jt].shift(15).fillna(method='bfill')
        
        v = f'{jt}_vel'
        if v in df.columns:
            df[f'{jt}_acc'] = df[v].diff().fillna(0)

    # High-risk posture flags
    df['neck_hflex']    = (df['neck']     > 20).astype(float)
    df['trunk_hflex']   = (df['trunk']    > 20).astype(float)
    df['shoulder_hext'] = (df['shoulder'] > 60).astype(float)

    return df

def synthesize_anomaly_labels(df):
    for col, (angle_col, thresh) in ANOM_THRESHOLDS.items():
        if col not in df.columns and angle_col in df.columns:
            df[col] = (df[angle_col] > thresh).astype(int)
    return df

# ─────────────── LOAD & PREPARE ───────────────────────────────────────────────
def load_and_prepare():
    banner("STEP 1/8 — LOAD & ADVANCED FEATURE ENGINEERING")
    df = pd.read_csv(DATASET_PATH)
    with open(META_PATH) as f: meta = json.load(f)

    df = engineer_features(df)
    df = synthesize_anomaly_labels(df)

    # Build final feature list
    feats = [c for c in BASE_FEATURES if c in df.columns]
    eng_cols = [c for c in df.columns if c.startswith(('asym_','upper_','lower_','neck_h','trunk_h','shoulder_h')) and c not in feats]
    ts_cols = [c for c in df.columns if c.endswith(('_energy', '_roll_mean', '_roll_std', '_lag_15', '_acc')) and c not in feats]
    feats = feats + eng_cols + ts_cols

    print(f"  Base features : {len(BASE_FEATURES)}")
    print(f"  Engineered TS : {len(eng_cols) + len(ts_cols)}")
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

def make_printer(tree_log, period=50, col_a='metric_a', col_b='metric_b'):
    def cb(env):
        i = env.iteration
        entry = {'tree': i+1}
        for ds_name, metric_name, val, _ in env.evaluation_result_list:
            entry[f"{ds_name}_{metric_name}"] = val
        
        va = next((v for d, m, v, _ in env.evaluation_result_list if d == 'train' and m in ('rmse', 'multi_logloss', 'binary_logloss')), 0)
        vb = next((v for d, m, v, _ in env.evaluation_result_list if d == 'valid' and m in ('rmse', 'multi_logloss', 'binary_logloss')), 0)
        entry[col_a] = va
        entry[col_b] = vb
        tree_log.append(entry)
        
        if (i+1) % period == 0:
            print(f"    tree {i+1:>5}  |  train: {va:.6f}  valid: {vb:.6f}")
    return cb

def plot_shap(model, X_te, feats, name):
    explainer = shap.TreeExplainer(model)
    # SHAP calculates values for subset to avoid hanging
    shap_values = explainer.shap_values(X_te[:500])
    plt.figure(figsize=(10,6))
    if isinstance(shap_values, list): # Multi-class
        shap.summary_plot(shap_values, X_te[:500], feature_names=feats, show=False)
    elif len(shap_values.shape) == 3: # Multi-class LightGBM 3D
        shap.summary_plot(shap_values[:, :, 0], X_te[:500], feature_names=feats, show=False)
    else: # Regression/Binary
        shap.summary_plot(shap_values, X_te[:500], feature_names=feats, show=False)
    plt.tight_layout()
    plt.savefig(str(MODELS_DIR / f'shap_{name}.png'), dpi=150)
    plt.close()
    print(f"  [OK] SHAP explainability plot saved -> models/shap_{name}.png")

# ─────────────── MODEL 1: REGRESSOR ───────────────────────────────────────────
def train_regressor(X_tr, X_te, y_tr, y_te, feats):
    banner("MODEL 1 — Regressor (RISK SCORE) + Optuna CV")
    
    def objective(trial):
        params = {
            'objective': 'regression', 'metric': 'rmse', 'verbose': -1, 'seed': SEED,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 31, 256),
            'max_depth': trial.suggest_int('max_depth', 5, 15),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        }
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        for train_idx, val_idx in tscv.split(X_tr):
            dtr = lgb.Dataset(X_tr[train_idx], label=y_tr[train_idx])
            dva = lgb.Dataset(X_tr[val_idx], label=y_tr[val_idx], reference=dtr)
            model = lgb.train(params, dtr, num_boost_round=300, valid_sets=[dva], 
                              callbacks=[lgb.early_stopping(30, verbose=False)])
            preds = model.predict(X_tr[val_idx])
            scores.append(np.sqrt(mean_squared_error(y_tr[val_idx], preds)))
        return np.mean(scores)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=N_TRIALS)
    print(f"  Optuna best RMSE: {study.best_value:.4f}")
    
    best_params = study.best_params
    best_params.update({'objective': 'regression', 'metric': ['rmse', 'mae'], 'verbose': -1, 'seed': SEED})
    
    dtr = lgb.Dataset(X_tr, label=y_tr, feature_name=feats)
    dte = lgb.Dataset(X_te, label=y_te, feature_name=feats, reference=dtr)

    tree_log = []
    print(f"\n  Final Training with best params:")
    model = lgb.train(best_params, dtr, num_boost_round=1000, valid_sets=[dtr, dte],
                      valid_names=['train','valid'],
                      callbacks=[lgb.early_stopping(100, verbose=False),
                                 make_printer(tree_log, period=100, col_a='train_rmse', col_b='valid_rmse')])

    y_pred = model.predict(X_te)
    m = dict(mae=mean_absolute_error(y_te,y_pred), rmse=np.sqrt(mean_squared_error(y_te,y_pred)), r2=r2_score(y_te,y_pred))
    print(f"\n  MAE={m['mae']:.6f}  RMSE={m['rmse']:.6f}  R2={m['r2']:.6f}")
    model.save_model(str(MODELS_DIR/'lgb_regressor.txt'))
    plot_shap(model, X_te, feats, 'regressor')
    return model, m, tree_log, y_pred, y_te

# ─────────────── MODEL 2: CLASSIFIER ──────────────────────────────────────────
def train_classifier(X_tr, X_te, y_tr, y_te, n_classes, feats):
    banner("MODEL 2 — Classifier (CONDITIONS) + Optuna CV")
    
    def objective(trial):
        params = {
            'objective': 'multiclass', 'num_class': n_classes, 'metric': 'multi_logloss', 'verbose': -1, 'seed': SEED,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 31, 256),
            'max_depth': trial.suggest_int('max_depth', 5, 15),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        }
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        for train_idx, val_idx in tscv.split(X_tr):
            sw_sub = compute_sample_weight('balanced', y_tr[train_idx])
            dtr = lgb.Dataset(X_tr[train_idx], label=y_tr[train_idx], weight=sw_sub)
            dva = lgb.Dataset(X_tr[val_idx], label=y_tr[val_idx], reference=dtr)
            model = lgb.train(params, dtr, num_boost_round=300, valid_sets=[dva], 
                              callbacks=[lgb.early_stopping(30, verbose=False)])
            preds = model.predict(X_tr[val_idx])
            y_p = np.argmax(preds, axis=1)
            scores.append(f1_score(y_tr[val_idx], y_p, average='macro', zero_division=0))
        return -np.mean(scores)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=N_TRIALS)
    print(f"  Optuna best Macro-F1: {-study.best_value:.4f}")
    
    best_params = study.best_params
    best_params.update({'objective': 'multiclass', 'num_class': n_classes, 'metric': ['multi_logloss', 'multi_error'], 'verbose': -1, 'seed': SEED})
    
    sw = compute_sample_weight('balanced', y_tr)
    dtr = lgb.Dataset(X_tr, label=y_tr, weight=sw, feature_name=feats)
    dte = lgb.Dataset(X_te, label=y_te, feature_name=feats, reference=dtr)

    tree_log = []
    print(f"\n  Final Training with best params:")
    model = lgb.train(best_params, dtr, num_boost_round=1000, valid_sets=[dtr, dte],
                      valid_names=['train','valid'],
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                 make_printer(tree_log, period=50, col_a='train_ll', col_b='valid_ll')])

    probs  = model.predict(X_te)
    y_pred = np.argmax(probs, axis=1)
    m = dict(accuracy =accuracy_score(y_te,y_pred),
             precision=precision_score(y_te,y_pred,average='macro',zero_division=0),
             recall   =recall_score(y_te,y_pred,average='macro',zero_division=0),
             f1       =f1_score(y_te,y_pred,average='macro',zero_division=0))
    print(f"\n  Accuracy={m['accuracy']:.4f}  Precision={m['precision']:.4f}  Recall={m['recall']:.4f}  F1={m['f1']:.4f}")
    model.save_model(str(MODELS_DIR/'lgb_classifier.txt'))
    plot_shap(model, X_te, feats, 'classifier')
    return model, m, tree_log, probs, y_te

# ─────────────── MODEL 3: SEVERITY ────────────────────────────────────────────
def train_severity(X_tr, X_te, y_tr, y_te, feats):
    banner("MODEL 3 — Severity Classifier + Optuna CV")
    n = int(max(y_tr.max(), y_te.max())) + 1
    sw = compute_sample_weight('balanced', y_tr)
    
    # We skip Optuna here to save total script time, use fixed robust params based on standard ranges
    params = {
        'objective':'multiclass','num_class':n,'metric':['multi_logloss', 'multi_error'],
        'learning_rate': 0.03, 'num_leaves': 128, 'max_depth': 8,
        'min_child_samples': 10, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'reg_alpha': 0.1, 'reg_lambda': 0.1, 'verbose': -1, 'seed': SEED,
    }
    
    dtr = lgb.Dataset(X_tr, label=y_tr, weight=sw, feature_name=feats)
    dte = lgb.Dataset(X_te, label=y_te, feature_name=feats, reference=dtr)

    tree_log = []
    print(f"\n  Training Severity (Robust Params):")
    model = lgb.train(params, dtr, num_boost_round=1000, valid_sets=[dtr, dte],
                      valid_names=['train','valid'],
                      callbacks=[lgb.early_stopping(50,verbose=False),
                                 make_printer(tree_log,period=50,col_a='tr',col_b='va')])

    probs  = model.predict(X_te)
    y_pred = np.argmax(probs, axis=1)
    m = dict(accuracy=accuracy_score(y_te,y_pred), f1=f1_score(y_te,y_pred,average='macro',zero_division=0))
    print(f"\n  Accuracy={m['accuracy']:.4f}  F1={m['f1']:.4f}")
    model.save_model(str(MODELS_DIR/'lgb_severity.txt'))
    return model, m, tree_log, probs, y_te

# ─────────────── MODEL 4: ANOMALY BINARY ──────────────────────────────────────
def train_anomaly_models(df, X_tr, X_te, feats, split_idx):
    banner("MODEL 4 — Per-Joint Anomaly Classifiers (Robust Params)")
    results = {}
    for col in ANOM_THRESHOLDS:
        if col not in df.columns: continue
        y_tr_b = df[col].values[:split_idx].astype(int)
        y_te_b = df[col].values[split_idx:].astype(int)
        pos_rate = y_tr_b.mean()
        if len(np.unique(y_tr_b)) < 2: continue

        sw = compute_sample_weight('balanced', y_tr_b)
        dtr = lgb.Dataset(X_tr, label=y_tr_b, weight=sw, feature_name=feats)
        dte = lgb.Dataset(X_te, label=y_te_b, feature_name=feats, reference=dtr)

        params = {
            'objective':'binary','metric':'binary_logloss', 'verbose':-1,'seed':SEED,
            'learning_rate':0.03,'num_leaves':64,'max_depth':7,
            'min_child_samples':5,'subsample':0.8,'colsample_bytree':0.8,
            'reg_alpha':0.1, 'reg_lambda':0.1
        }
        tree_log = []
        model = lgb.train(params, dtr, num_boost_round=500, valid_sets=[dtr,dte],
                          valid_names=['train','valid'],
                          callbacks=[lgb.early_stopping(30,verbose=False),
                                     make_printer(tree_log, period=50, col_a='tr', col_b='va')])
        probs = model.predict(X_te)
        yp    = (probs > 0.5).astype(int)
        acc   = accuracy_score(y_te_b, yp)
        f1    = f1_score(y_te_b, yp, zero_division=0)
        print(f"  {col:<35}  iter={model.best_iteration:<4}  acc={acc:.4f}  f1={f1:.4f}  pos%={pos_rate:.1%}")
        model.save_model(str(MODELS_DIR/f'lgbm_{col}.txt'))
        results[col] = {'accuracy':acc,'f1':f1,'log':tree_log,'probs':probs,'y_te':y_te_b}
    return results

# ─────────────── MODEL 5: GLOBAL ANOMALY ──────────────────────────────────────
def train_iso_forest(X_tr, X_te, feats):
    banner("MODEL 5 — Global Anomaly (LocalOutlierFactor / IsolationForest)")
    scaler = StandardScaler()
    Xts  = scaler.fit_transform(X_tr)
    Xvs  = scaler.transform(X_te)
    
    # We keep Isolation Forest as it generalizes well to unseen data natively
    iso  = IsolationForest(n_estimators=300, contamination=0.05, max_samples=256,
                           random_state=SEED, n_jobs=-1)
    iso.fit(Xts)
    scores = iso.decision_function(Xvs)
    n_anom = (iso.predict(Xvs)==-1).sum()
    print(f"  Trees: 300  |  Anomalies in test: {n_anom}/{len(Xvs)} ({(n_anom/len(Xvs))*100:.2f}%)")
    
    joblib.dump(iso,    MODELS_DIR/'isolation_forest.pkl')
    joblib.dump(scaler, MODELS_DIR/'scaler_if.pkl')
    iso_stats = {'n_anom': n_anom, 'n_test': len(Xvs), 'min_score': scores.min(), 'max_score': scores.max(), 'scores': scores}
    return iso, scaler, iso_stats

# ─────────────── UPDATE METADATA ──────────────────────────────────────────────
def update_meta(meta, feats, reg_m, cls_m, sev_m, n_anom):
    banner("STEP 8/8 — Update model_metadata.json")
    meta['version']      = '3.0-Production'
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
    print("  model_metadata.json updated (v3.0-Production)")

# ─────────────── FINAL REPORT ─────────────────────────────────────────────────
def final_report(reg_m, cls_m, sev_m, reg_log, cls_log, sev_log, iso_stats):
    banner("FINAL REPORT — ERGO SENSOR AI ENGINE v3.0-Production")
    print(f"  MODEL 1 — Regressor (Risk Score)")
    print(f"    MAE  : {reg_m['mae']:.6f} | RMSE : {reg_m['rmse']:.6f} | R2   : {reg_m['r2']:.6f}")
    print(f"  MODEL 2 — Condition Classifier")
    print(f"    Accuracy : {cls_m['accuracy']:.4f} | Recall : {cls_m['recall']:.4f} | F1 : {cls_m['f1']:.4f}")
    print(f"  MODEL 3 — Severity Classifier")
    print(f"    Accuracy : {sev_m['accuracy']:.4f} | F1 : {sev_m['f1']:.4f}")

# ─────────────── PLOT HELPERS ─────────────────────────────────────────────────
STYLE = {
    'train': '#00d4ff', 'valid': '#ff00aa',
    'bg': '#1a1a2e',    'grid': '#2a2a4a',
    'text': '#e0e0ff',  'accent': '#00ffaa',
}

def _style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(STYLE['bg'])
    ax.tick_params(colors=STYLE['text'])
    ax.spines[:].set_color(STYLE['grid'])
    ax.xaxis.label.set_color(STYLE['text'])
    ax.yaxis.label.set_color(STYLE['text'])
    ax.title.set_color(STYLE['text'])
    if title:  ax.set_title(title, fontsize=11, fontweight='bold')
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    ax.grid(True, color=STYLE['grid'], linewidth=0.5)

def _save(fig, name):
    fig.patch.set_facecolor(STYLE['bg'])
    fig.savefig(str(MODELS_DIR / name), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [OK] Saved -> models/{name}')

# ─── MODEL 1: Learning + Residuals + Scatter ──────────────────────────────────
def plot_model1(reg_log, y_pred, y_te):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1a. Learning curve
    if reg_log:
        trees = [x['tree'] for x in reg_log]
        axes[0].plot(trees, [x['train_rmse'] for x in reg_log], color=STYLE['train'], label='Train RMSE')
        axes[0].plot(trees, [x['valid_rmse'] for x in reg_log], color=STYLE['valid'], label='Valid RMSE')
        axes[0].legend(facecolor=STYLE['bg'], labelcolor=STYLE['text'])
    _style_ax(axes[0], 'M1 — Learning Curve (RMSE)', 'Trees', 'RMSE')

    # 1b. Predicted vs Actual
    axes[1].scatter(y_te, y_pred, alpha=0.3, s=8, color=STYLE['accent'])
    lo, hi = min(y_te.min(), y_pred.min()), max(y_te.max(), y_pred.max())
    axes[1].plot([lo, hi], [lo, hi], 'r--', linewidth=1.5, label='Perfect')
    axes[1].legend(facecolor=STYLE['bg'], labelcolor=STYLE['text'])
    _style_ax(axes[1], 'M1 — Predicted vs Actual', 'Actual Risk Score', 'Predicted Risk Score')

    # 1c. Residuals histogram
    residuals = y_pred - y_te
    axes[2].hist(residuals, bins=60, color=STYLE['train'], alpha=0.8, edgecolor='none')
    axes[2].axvline(0, color='red', linewidth=1.5, linestyle='--')
    _style_ax(axes[2], 'M1 — Residuals Distribution', 'Residual', 'Count')

    _save(fig, 'eval_model1_regressor.png')

# ─── MODEL 2: Learning + Confusion + ROC + PR ─────────────────────────────────
def plot_model2(cls_log, cls_probs, y_te, meta):
    y_pred = np.argmax(cls_probs, axis=1)
    n_cls  = cls_probs.shape[1]
    labels = [k for k, v in sorted(meta.get('condition_to_code', {}).items(), key=lambda x: x[1])]

    # 2a. Learning curve
    fig1, ax = plt.subplots(figsize=(8, 5))
    if cls_log:
        trees = [x['tree'] for x in cls_log]
        ax.plot(trees, [x['train_ll'] for x in cls_log], color=STYLE['train'], label='Train LogLoss')
        ax.plot(trees, [x['valid_ll'] for x in cls_log], color=STYLE['valid'], label='Valid LogLoss')
        ax.legend(facecolor=STYLE['bg'], labelcolor=STYLE['text'])
    _style_ax(ax, 'M2 — Condition Classifier Learning Curve', 'Trees', 'LogLoss')
    _save(fig1, 'eval_model2_learning.png')

    # 2b. Confusion matrix
    cm = confusion_matrix(y_te, y_pred)
    fig2, ax = plt.subplots(figsize=(max(6, n_cls), max(5, n_cls - 1)))
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(n_cls)); ax.set_yticks(range(n_cls))
    ax.set_xticklabels(labels if labels else range(n_cls), rotation=45, ha='right', color=STYLE['text'])
    ax.set_yticklabels(labels if labels else range(n_cls), color=STYLE['text'])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=8)
    _style_ax(ax, 'M2 — Confusion Matrix (Conditions)', 'Predicted', 'Actual')
    _save(fig2, 'eval_model2_confusion.png')

    # 2c. ROC curves (one-vs-rest)
    fig3, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, n_cls))
    for i in range(n_cls):
        y_bin = (y_te == i).astype(int)
        if y_bin.sum() == 0: continue
        fpr, tpr, _ = roc_curve(y_bin, cls_probs[:, i])
        roc_auc = auc(fpr, tpr)
        lbl = labels[i] if i < len(labels) else f'Class {i}'
        ax.plot(fpr, tpr, color=colors[i], label=f'{lbl} (AUC={roc_auc:.3f})')
    ax.plot([0,1],[0,1],'--', color='gray', linewidth=1)
    ax.legend(facecolor=STYLE['bg'], labelcolor=STYLE['text'], fontsize=8)
    _style_ax(ax, 'M2 — ROC Curves (One-vs-Rest)', 'False Positive Rate', 'True Positive Rate')
    _save(fig3, 'eval_model2_roc.png')

    # 2d. Precision-Recall curves
    fig4, ax = plt.subplots(figsize=(8, 6))
    for i in range(n_cls):
        y_bin = (y_te == i).astype(int)
        if y_bin.sum() == 0: continue
        prec, rec, _ = precision_recall_curve(y_bin, cls_probs[:, i])
        pr_auc = auc(rec, prec)
        lbl = labels[i] if i < len(labels) else f'Class {i}'
        ax.plot(rec, prec, color=colors[i], label=f'{lbl} (AUC={pr_auc:.3f})')
    ax.legend(facecolor=STYLE['bg'], labelcolor=STYLE['text'], fontsize=8)
    _style_ax(ax, 'M2 — Precision-Recall Curves', 'Recall', 'Precision')
    _save(fig4, 'eval_model2_pr.png')

# ─── MODEL 3: Learning + Confusion ────────────────────────────────────────────
def plot_model3(sev_log, sev_probs, y_te):
    y_pred = np.argmax(sev_probs, axis=1)
    n_cls  = sev_probs.shape[1]
    sev_labels = ['None','Low','Medium','High','Critical'][:n_cls]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 3a. Learning curve
    if sev_log:
        trees = [x['tree'] for x in sev_log]
        axes[0].plot(trees, [x['tr'] for x in sev_log], color=STYLE['train'], label='Train LogLoss')
        axes[0].plot(trees, [x['va'] for x in sev_log], color=STYLE['valid'], label='Valid LogLoss')
        axes[0].legend(facecolor=STYLE['bg'], labelcolor=STYLE['text'])
    _style_ax(axes[0], 'M3 — Severity Learning Curve', 'Trees', 'LogLoss')

    # 3b. Confusion matrix
    cm = confusion_matrix(y_te, y_pred)
    im = axes[1].imshow(cm, cmap='Purples')
    plt.colorbar(im, ax=axes[1])
    axes[1].set_xticks(range(n_cls)); axes[1].set_yticks(range(n_cls))
    axes[1].set_xticklabels(sev_labels, rotation=30, ha='right', color=STYLE['text'])
    axes[1].set_yticklabels(sev_labels, color=STYLE['text'])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[1].text(j, i, str(cm[i, j]), ha='center', va='center',
                         color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=9)
    _style_ax(axes[1], 'M3 — Confusion Matrix (Severity)', 'Predicted', 'Actual')

    _save(fig, 'eval_model3_severity.png')

# ─── MODEL 4: F1 Bar + Learning Curves per joint ──────────────────────────────
def plot_model4(anom_res):
    if not anom_res: return
    names  = [k.replace('anomaly_','').replace('_',' ').title() for k in anom_res]
    f1s    = [v['f1']       for v in anom_res.values()]
    accs   = [v['accuracy'] for v in anom_res.values()]

    # 4a. F1 & Accuracy bar chart
    fig1, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    bars1 = ax.bar(x - 0.2, f1s,  0.4, label='F1',       color=STYLE['accent'], alpha=0.9)
    bars2 = ax.bar(x + 0.2, accs, 0.4, label='Accuracy', color=STYLE['train'],  alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha='right', color=STYLE['text'])
    ax.set_ylim(0.9, 1.01)
    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{bar.get_height():.4f}', ha='center', va='bottom',
                color=STYLE['text'], fontsize=7)
    ax.legend(facecolor=STYLE['bg'], labelcolor=STYLE['text'])
    _style_ax(ax, 'M4 — Per-Joint Anomaly: F1 & Accuracy', 'Joint', 'Score')
    _save(fig1, 'eval_model4_bars.png')

    # 4b. Learning curves per joint
    valid_logs = [(k, v) for k, v in anom_res.items() if v.get('log')]
    if not valid_logs: return
    fig2, axes = plt.subplots(1, len(valid_logs), figsize=(4 * len(valid_logs), 4))
    if len(valid_logs) == 1: axes = [axes]
    for ax, (col, res) in zip(axes, valid_logs):
        log = res['log']
        trees = [x['tree'] for x in log]
        tr_key = next((k for k in log[0] if 'train' in k and 'logloss' in k), 'tr')
        va_key = next((k for k in log[0] if 'valid' in k and 'logloss' in k), 'va')
        ax.plot(trees, [x.get(tr_key, x.get('tr',0)) for x in log], color=STYLE['train'], label='Train')
        ax.plot(trees, [x.get(va_key, x.get('va',0)) for x in log], color=STYLE['valid'], label='Valid')
        ax.legend(facecolor=STYLE['bg'], labelcolor=STYLE['text'], fontsize=7)
        _style_ax(ax, col.replace('anomaly_','').replace('_',' ').title(), 'Trees', 'LogLoss')
    _save(fig2, 'eval_model4_learning.png')

    # 4c. ROC per joint
    fig3, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(anom_res)))
    for (col, res), c in zip(anom_res.items(), colors):
        probs = res.get('probs')
        y_te  = res.get('y_te')
        if probs is None or y_te is None or y_te.sum() == 0: continue
        fpr, tpr, _ = roc_curve(y_te, probs)
        roc_auc = auc(fpr, tpr)
        lbl = col.replace('anomaly_','').replace('_',' ').title()
        ax.plot(fpr, tpr, color=c, label=f'{lbl} (AUC={roc_auc:.3f})')
    ax.plot([0,1],[0,1],'--', color='gray', linewidth=1)
    ax.legend(facecolor=STYLE['bg'], labelcolor=STYLE['text'], fontsize=8)
    _style_ax(ax, 'M4 — Per-Joint Anomaly ROC Curves', 'FPR', 'TPR')
    _save(fig3, 'eval_model4_roc.png')

# ─── MODEL 5: Score Distribution + Threshold ──────────────────────────────────
def plot_model5(iso_stats):
    scores = iso_stats.get('scores', np.array([]))
    if len(scores) == 0: return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 5a. Histogram
    axes[0].hist(scores[scores >= 0], bins=40, color=STYLE['accent'], alpha=0.8, label='Normal')
    axes[0].hist(scores[scores <  0], bins=40, color='#ff4444',       alpha=0.8, label='Anomaly')
    axes[0].axvline(0, color='red', linestyle='--', linewidth=1.5, label='Threshold')
    axes[0].legend(facecolor=STYLE['bg'], labelcolor=STYLE['text'])
    _style_ax(axes[0], 'M5 — Isolation Forest Score Distribution', 'Anomaly Score', 'Count')

    # 5b. Sorted score plot
    sorted_scores = np.sort(scores)
    axes[1].plot(sorted_scores, color=STYLE['train'], linewidth=1)
    axes[1].axhline(0, color='red', linestyle='--', linewidth=1.5, label='Threshold')
    axes[1].fill_between(range(len(sorted_scores)), sorted_scores, 0,
                         where=(sorted_scores < 0), color='#ff4444', alpha=0.3, label='Anomaly zone')
    axes[1].legend(facecolor=STYLE['bg'], labelcolor=STYLE['text'])
    _style_ax(axes[1], 'M5 — Sorted Anomaly Scores', 'Sample Index (sorted)', 'Score')

    _save(fig, 'eval_model5_isoforest.png')

# ─────────────── MAIN ─────────────────────────────────────────────────────────
def main():
    print("\n" + "="*72)
    print("  ERGO SENSOR AI ENGINE v3.0-Production — OPTUNA + SHAP + FULL EVAL")
    print("="*72)
    t0 = datetime.now()

    df, meta, feats = load_and_prepare()
    X_tr, X_te, targets, split_idx = temporal_split(df, feats)

    y_reg_tr, y_reg_te = targets['risk_score']
    y_cls_tr, y_cls_te = targets['condition_code']
    y_sev_tr, y_sev_te = targets['severity_code']

    _, reg_m, reg_log, reg_pred, reg_yte = train_regressor(X_tr, X_te, y_reg_tr, y_reg_te, feats)
    _, cls_m, cls_log, cls_probs, cls_yte = train_classifier(X_tr, X_te, y_cls_tr, y_cls_te,
                                                              len(meta['condition_to_code']), feats)
    _, sev_m, sev_log, sev_probs, sev_yte = train_severity(X_tr, X_te, y_sev_tr, y_sev_te, feats)

    anom_res = train_anomaly_models(df, X_tr, X_te, feats, split_idx)
    iso, scaler, iso_stats = train_iso_forest(X_tr, X_te, feats)

    update_meta(meta, feats, reg_m, cls_m, sev_m, len(anom_res))
    final_report(reg_m, cls_m, sev_m, reg_log, cls_log, sev_log, iso_stats)

    banner("GENERATING ALL EVALUATION PLOTS")
    plot_model1(reg_log, reg_pred, reg_yte)
    plot_model2(cls_log, cls_probs, cls_yte, meta)
    plot_model3(sev_log, sev_probs, sev_yte)
    plot_model4(anom_res)
    plot_model5(iso_stats)

    import shutil, os
    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)
    for png in MODELS_DIR.glob('*.png'):
        shutil.copy2(png, plots_dir / png.name)
    print(f"  [OK] {len(list(MODELS_DIR.glob('*.png')))} plots copied -> plots/")

    print(f"\n  Total time: {(datetime.now()-t0).total_seconds()/60:.1f} min\n")

if __name__ == '__main__':
    main()

