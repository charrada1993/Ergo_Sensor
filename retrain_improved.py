"""
=============================================================================
RETRAINING AMÉLIORÉ - ERGO SENSOR AI ENGINE v2.1
=============================================================================
Auteur : Ingénieur IA Senior
Date   : 2026
Version: 2.1 (38 features, hyperparams optimisés)

✅ Ce script :
   1. Charge le dataset enrichi (38 features)
   2. Réentraîne les modèles LightGBM avec de meilleurs hyperparamètres
   3. Affiche les résultats arbre par arbre (tree results)
   4. Affiche les métriques époque par époque (epoch results)
   5. Sauvegarde les modèles + met à jour model_metadata.json
=============================================================================
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pandas as pd
import numpy as np
import json
import joblib
import warnings
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import shap

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET_PATH  = 'dataset_TMS_enriched.csv'
MODELS_DIR    = Path('models')
META_PATH     = MODELS_DIR / 'model_metadata.json'
TEST_SIZE     = 0.20
RANDOM_STATE  = 42

# 38 features from model_metadata.json
FEATURE_COLS = [
    'neck', 'trunk', 'shoulder', 'elbow', 'wrist', 'hip', 'knee',
    'r_shoulder', 'l_shoulder', 'r_elbow', 'l_elbow',
    'r_wrist', 'l_wrist', 'r_hip', 'l_hip', 'r_knee', 'l_knee',
    'neck_vel', 'trunk_vel', 'shoulder_vel', 'elbow_vel', 'wrist_vel',
    'hip_vel', 'knee_vel',
    'neck_duration', 'trunk_duration', 'shoulder_duration',
    'elbow_duration', 'wrist_duration', 'hip_duration', 'knee_duration',
    'neck_freq', 'trunk_freq', 'shoulder_freq',
    'elbow_freq', 'wrist_freq', 'hip_freq', 'knee_freq',
]

ANOMALY_COLS = [
    'anomaly_neck_hyperflex',
    'anomaly_shoulder_overext',
    'anomaly_wrist_strain',
    'anomaly_trunk_torsion',
    'anomaly_elbow_hyperext',
]

# ============================================================================
# UTILITY
# ============================================================================

def banner(title, char='='):
    line = char * 80
    print(f"\n{line}")
    print(f"  {title}")
    print(line)

def check_and_fill_cols(df, cols):
    """Fill any missing feature columns with 0."""
    for c in cols:
        if c not in df.columns:
            print(f"  [WARN] Column '{c}' missing → filled with 0")
            df[c] = 0.0
    return df

# ============================================================================
# SECTION 1 : CHARGEMENT
# ============================================================================

def load_data():
    banner("ÉTAPE 1 / 8 — CHARGEMENT DU DATASET")
    df = pd.read_csv(DATASET_PATH)
    print(f"  ✅ {len(df):,} samples  |  {len(df.columns)} colonnes")

    # Load metadata for mappings
    with open(META_PATH) as f:
        meta = json.load(f)

    df = check_and_fill_cols(df, FEATURE_COLS)

    # Encode target: condition_code
    condition_to_code = meta['condition_to_code']
    if 'condition_code' not in df.columns:
        if 'main_condition' in df.columns:
            df['condition_code'] = df['main_condition'].map(condition_to_code).fillna(11).astype(int)
        else:
            df['condition_code'] = 11

    # severity_code
    severity_to_code = meta['severity_to_code']
    if 'severity_code' not in df.columns:
        if 'severity' in df.columns:
            df['severity_code'] = df['severity'].map(severity_to_code).fillna(0).astype(int)
        else:
            df['severity_code'] = 0

    # risk_score
    if 'risk_score' not in df.columns:
        df['risk_score'] = np.random.uniform(0, 1, len(df))

    return df, meta

# ============================================================================
# SECTION 2 : SPLIT
# ============================================================================

def split_data(df):
    banner("ÉTAPE 2 / 8 — SPLIT TEMPOREL TRAIN / TEST")

    # Use all available feature cols that are in df
    feats = [c for c in FEATURE_COLS if c in df.columns]

    X = df[feats].values.astype(np.float32)
    y_reg = df['risk_score'].values.astype(np.float32)
    y_cls = df['condition_code'].values.astype(int)
    y_sev = df['severity_code'].values.astype(int)

    # Temporal split (no shuffle)
    split_idx = int(len(df) * (1 - TEST_SIZE))
    X_tr, X_te = X[:split_idx], X[split_idx:]
    y_reg_tr, y_reg_te = y_reg[:split_idx], y_reg[split_idx:]
    y_cls_tr, y_cls_te = y_cls[:split_idx], y_cls[split_idx:]
    y_sev_tr, y_sev_te = y_sev[:split_idx], y_sev[split_idx:]

    print(f"  Train : {X_tr.shape[0]:,} samples")
    print(f"  Test  : {X_te.shape[0]:,} samples")
    print(f"  Features utilisées : {X_tr.shape[1]}")

    return (X_tr, X_te, y_reg_tr, y_reg_te,
            y_cls_tr, y_cls_te, y_sev_tr, y_sev_te, feats)

# ============================================================================
# SECTION 3 : MODÈLE 1 — LightGBM Régression (RISK SCORE)
# ============================================================================

def train_regressor(X_tr, X_te, y_tr, y_te, feature_names):
    banner("MODÈLE 1 — LightGBM Régression (RISK SCORE)")

    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_names)
    dtest  = lgb.Dataset(X_te, label=y_te, feature_name=feature_names, reference=dtrain)

    params = {
        'objective':        'regression',
        'metric':           'rmse',
        'learning_rate':    0.05,       # faster convergence than 0.01
        'num_leaves':       63,          # more complex trees
        'max_depth':        7,
        'min_child_samples': 20,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq':     5,
        'reg_alpha':        0.1,         # L1
        'reg_lambda':       0.2,         # L2
        'verbose':          -1,
        'seed':             RANDOM_STATE,
    }

    print("\n  📊 TREE RESULTS (every 50 trees):")
    print(f"  {'Tree':>6}  {'Train RMSE':>12}  {'Valid RMSE':>12}")
    print("  " + "-" * 36)

    evals_result = {}
    tree_log = []

    class TreePrinterCallback:
        def __init__(self, period=50):
            self.period = period
        def __call__(self, env):
            i = env.iteration
            tr_rmse = env.evaluation_result_list[0][2]
            va_rmse = env.evaluation_result_list[1][2]
            tree_log.append({'tree': i+1, 'train_rmse': tr_rmse, 'valid_rmse': va_rmse})
            if (i + 1) % self.period == 0:
                print(f"  {i+1:>6}  {tr_rmse:>12.6f}  {va_rmse:>12.6f}")

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dtrain, dtest],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=80, verbose=False),
            lgb.record_evaluation(evals_result),
            TreePrinterCallback(period=50),
        ],
    )

    print(f"\n  ✅ Best iteration: {model.best_iteration}")

    y_pred = model.predict(X_te)
    mae  = mean_absolute_error(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    r2   = r2_score(y_te, y_pred)

    print(f"\n  📈 FINAL METRICS — Regressor:")
    print(f"     MAE  : {mae:.6f}")
    print(f"     RMSE : {rmse:.6f}")
    print(f"     R²   : {r2:.6f}")

    model.save_model(str(MODELS_DIR / 'lgb_regressor.txt'))
    print(f"\n  💾 Saved → models/lgb_regressor.txt")

    return model, {'mae': mae, 'rmse': rmse, 'r2': r2}, tree_log

# ============================================================================
# SECTION 4 : MODÈLE 2 — LightGBM Classification (CONDITIONS)
# ============================================================================

def train_classifier(X_tr, X_te, y_tr, y_te, n_classes, feature_names):
    banner("MODÈLE 2 — LightGBM Classification (CONDITIONS)")

    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_names)
    dtest  = lgb.Dataset(X_te, label=y_te, feature_name=feature_names, reference=dtrain)

    params = {
        'objective':         'multiclass',
        'num_class':         n_classes,
        'metric':            'multi_logloss',
        'learning_rate':     0.05,
        'num_leaves':        63,
        'max_depth':         7,
        'min_child_samples': 20,
        'feature_fraction':  0.8,
        'bagging_fraction':  0.8,
        'bagging_freq':      5,
        'reg_alpha':         0.1,
        'reg_lambda':        0.2,
        'verbose':           -1,
        'seed':              RANDOM_STATE,
    }

    print("\n  📊 TREE RESULTS (every 50 trees):")
    print(f"  {'Tree':>6}  {'Train LogLoss':>14}  {'Valid LogLoss':>14}")
    print("  " + "-" * 40)

    evals_result = {}
    tree_log = []

    class TreePrinterCls:
        def __init__(self, period=50):
            self.period = period
        def __call__(self, env):
            i = env.iteration
            tr_loss = env.evaluation_result_list[0][2]
            va_loss = env.evaluation_result_list[1][2]
            tree_log.append({'tree': i+1, 'train_logloss': tr_loss, 'valid_logloss': va_loss})
            if (i + 1) % self.period == 0:
                print(f"  {i+1:>6}  {tr_loss:>14.6f}  {va_loss:>14.6f}")

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dtrain, dtest],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=80, verbose=False),
            lgb.record_evaluation(evals_result),
            TreePrinterCls(period=50),
        ],
    )

    print(f"\n  ✅ Best iteration: {model.best_iteration}")

    probs = model.predict(X_te)           # shape (N, n_classes)
    y_pred = np.argmax(probs, axis=1)

    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_te, y_pred,    average='macro', zero_division=0)
    f1   = f1_score(y_te, y_pred,        average='macro', zero_division=0)

    print(f"\n  📈 FINAL METRICS — Classifier:")
    print(f"     Accuracy  : {acc:.6f}")
    print(f"     Precision : {prec:.6f}")
    print(f"     Recall    : {rec:.6f}")
    print(f"     F1-Score  : {f1:.6f}")

    model.save_model(str(MODELS_DIR / 'lgb_classifier.txt'))
    print(f"\n  💾 Saved → models/lgb_classifier.txt")

    return model, {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}, tree_log

# ============================================================================
# SECTION 5 : MODÈLE 3 — LightGBM Severity
# ============================================================================

def train_severity(X_tr, X_te, y_tr, y_te, feature_names):
    banner("MODÈLE 3 — LightGBM Classification (SEVERITY)")

    n_classes = int(max(np.max(y_tr), np.max(y_te))) + 1

    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_names)
    dtest  = lgb.Dataset(X_te, label=y_te, feature_name=feature_names, reference=dtrain)

    params = {
        'objective':         'multiclass',
        'num_class':         n_classes,
        'metric':            'multi_logloss',
        'learning_rate':     0.05,
        'num_leaves':        31,
        'max_depth':         6,
        'min_child_samples': 20,
        'feature_fraction':  0.8,
        'bagging_fraction':  0.8,
        'bagging_freq':      5,
        'verbose':           -1,
        'seed':              RANDOM_STATE,
    }

    print("\n  📊 TREE RESULTS (every 50 trees):")
    print(f"  {'Tree':>6}  {'Train LogLoss':>14}  {'Valid LogLoss':>14}")
    print("  " + "-" * 40)

    evals_result = {}
    tree_log = []

    class TreePrinterSev:
        def __init__(self, period=50):
            self.period = period
        def __call__(self, env):
            i = env.iteration
            tr_loss = env.evaluation_result_list[0][2]
            va_loss = env.evaluation_result_list[1][2]
            tree_log.append({'tree': i+1, 'train_logloss': tr_loss, 'valid_logloss': va_loss})
            if (i + 1) % self.period == 0:
                print(f"  {i+1:>6}  {tr_loss:>14.6f}  {va_loss:>14.6f}")

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dtrain, dtest],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=80, verbose=False),
            lgb.record_evaluation(evals_result),
            TreePrinterSev(period=50),
        ],
    )

    print(f"\n  ✅ Best iteration: {model.best_iteration}")

    probs = model.predict(X_te)
    y_pred = np.argmax(probs, axis=1)
    acc = accuracy_score(y_te, y_pred)
    f1  = f1_score(y_te, y_pred, average='macro', zero_division=0)

    print(f"\n  📈 FINAL METRICS — Severity:")
    print(f"     Accuracy : {acc:.6f}")
    print(f"     F1-Score : {f1:.6f}")

    model.save_model(str(MODELS_DIR / 'lgb_severity.txt'))
    print(f"\n  💾 Saved → models/lgb_severity.txt")

    return model, {'accuracy': acc, 'f1': f1}, tree_log

# ============================================================================
# SECTION 6 : ANOMALY MODELS (5 articulations)
# ============================================================================

def train_anomaly_models(df, X_tr, X_te, feature_names):
    banner("MODÈLE 4 — ANOMALY CLASSIFIERS (5 articulations)")

    anomaly_results = {}
    thresholds = {}

    for col in ANOMALY_COLS:
        if col not in df.columns:
            print(f"  [SKIP] {col} not in dataset")
            continue

        split_idx = int(len(df) * (1 - TEST_SIZE))
        y_tr_bin = df[col].values[:split_idx].astype(int)
        y_te_bin = df[col].values[split_idx:].astype(int)

        # Skip if only one class
        if len(np.unique(y_tr_bin)) < 2:
            print(f"  [SKIP] {col} — only one class")
            continue

        # Store threshold (from dataset metadata)
        thresh_col = col.replace('anomaly_', '')
        if thresh_col in df.columns:
            thresholds[col] = float(np.percentile(df[thresh_col].dropna(), 90))

        dtrain = lgb.Dataset(X_tr, label=y_tr_bin, feature_name=feature_names)
        dtest  = lgb.Dataset(X_te, label=y_te_bin, feature_name=feature_names, reference=dtrain)

        params = {
            'objective':         'binary',
            'metric':            'binary_logloss',
            'learning_rate':     0.05,
            'num_leaves':        31,
            'max_depth':         5,
            'min_child_samples': 10,
            'feature_fraction':  0.7,
            'bagging_fraction':  0.8,
            'bagging_freq':      5,
            'verbose':           -1,
            'seed':              RANDOM_STATE,
            'is_unbalance':      True,
        }

        print(f"\n  ─── {col} ───")
        print(f"  {'Tree':>6}  {'Valid BinLoss':>14}")

        evals_result = {}
        tree_log_a = []

        class AnomalyPrinter:
            def __init__(self, col_name, period=50):
                self.col_name = col_name
                self.period = period
            def __call__(self, env):
                i = env.iteration
                va_loss = env.evaluation_result_list[1][2]
                tree_log_a.append({'tree': i+1, 'valid_logloss': va_loss})
                if (i + 1) % self.period == 0:
                    print(f"  {i+1:>6}  {va_loss:>14.6f}")

        model_a = lgb.train(
            params,
            dtrain,
            num_boost_round=500,
            valid_sets=[dtrain, dtest],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.record_evaluation(evals_result),
                AnomalyPrinter(col, period=50),
            ],
        )

        probs = model_a.predict(X_te)
        y_pred_bin = (probs > 0.5).astype(int)
        acc = accuracy_score(y_te_bin, y_pred_bin)
        f1  = f1_score(y_te_bin, y_pred_bin, zero_division=0)

        print(f"  ✅ Best iter: {model_a.best_iteration}  |  Acc: {acc:.4f}  F1: {f1:.4f}")

        model_a.save_model(str(MODELS_DIR / f'lgbm_{col}.txt'))
        anomaly_results[col] = {'accuracy': acc, 'f1': f1}

    return anomaly_results

# ============================================================================
# SECTION 7 : ISOLATION FOREST
# ============================================================================

def train_isolation_forest(X_tr, X_te, feature_names):
    banner("MODÈLE 5 — ISOLATION FOREST (anomalie globale)")

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    iso = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    iso.fit(X_tr_s)

    scores = iso.decision_function(X_te_s)
    n_anomalies = np.sum(iso.predict(X_te_s) == -1)

    print(f"  ✅ Isolation Forest trained ({iso.n_estimators} trees)")
    print(f"     Anomalies detected in test set: {n_anomalies:,} / {len(X_te):,}")
    print(f"     Score range: [{scores.min():.4f}, {scores.max():.4f}]")

    joblib.dump(iso,    MODELS_DIR / 'isolation_forest.pkl')
    joblib.dump(scaler, MODELS_DIR / 'scaler_if.pkl')
    print(f"  💾 Saved → models/isolation_forest.pkl + scaler_if.pkl")

    return iso, scaler

# ============================================================================
# SECTION 8 : UPDATE METADATA
# ============================================================================

def update_metadata(meta, reg_metrics, cls_metrics, sev_metrics, feature_names, anomaly_results):
    banner("ÉTAPE 8 / 8 — MISE À JOUR model_metadata.json")

    meta['version']    = '2.1'
    meta['created']    = datetime.utcnow().isoformat()
    meta['n_features'] = len(feature_names)
    meta['feature_cols'] = feature_names
    meta['if_features']  = feature_names

    meta['metrics'] = {
        'LightGBM_Regression': {
            'mae':  round(reg_metrics['mae'],  6),
            'rmse': round(reg_metrics['rmse'], 6),
            'r2':   round(reg_metrics['r2'],   6),
        },
        'LightGBM_Classifier': {
            'accuracy':  round(cls_metrics['accuracy'],  6),
            'precision': round(cls_metrics['precision'], 6),
            'recall':    round(cls_metrics['recall'],    6),
            'f1':        round(cls_metrics['f1'],        6),
        },
        'LightGBM_Severity': {
            'accuracy': round(sev_metrics['accuracy'], 6),
            'f1':       round(sev_metrics['f1'],       6),
        },
        'IsolationForest': {'trained': 1.0},
        'Anomaly_Models':  {'count': float(len(anomaly_results))},
    }

    with open(META_PATH, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"  ✅ model_metadata.json updated (v{meta['version']})")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

def print_summary(reg_metrics, cls_metrics, sev_metrics,
                  reg_tree_log, cls_tree_log, sev_tree_log):
    banner("RAPPORT FINAL — TOUS LES MODÈLES", char='*')

    print(f"""
  +─────────────────────────────────────────────────────────+
  |          ERGO SENSOR AI ENGINE v2.1 -- RESULTATS        |
  +─────────────────────────────────────────────────────────+

  📊 MODÈLE 1 : LightGBM Régression (Risk Score)
     MAE       : {reg_metrics['mae']:.6f}
     RMSE      : {reg_metrics['rmse']:.6f}
     R²        : {reg_metrics['r2']:.6f}
     Trees run : {len(reg_tree_log)}

  📊 MODÈLE 2 : LightGBM Classification (Conditions)
     Accuracy  : {cls_metrics['accuracy']:.6f}
     Precision : {cls_metrics['precision']:.6f}
     Recall    : {cls_metrics['recall']:.6f}
     F1-Score  : {cls_metrics['f1']:.6f}
     Trees run : {len(cls_tree_log)}

  📊 MODÈLE 3 : LightGBM Severity
     Accuracy  : {sev_metrics['accuracy']:.6f}
     F1-Score  : {sev_metrics['f1']:.6f}
     Trees run : {len(sev_tree_log)}
""")

    # Tree convergence summary for regressor
    if reg_tree_log:
        first = reg_tree_log[0]
        last  = reg_tree_log[-1]
        print(f"  📉 REGRESSOR CONVERGENCE:")
        print(f"     Tree   1 → valid RMSE: {first['valid_rmse']:.6f}")
        print(f"     Tree {last['tree']:>3} → valid RMSE: {last['valid_rmse']:.6f}")
        improvement = (first['valid_rmse'] - last['valid_rmse']) / first['valid_rmse'] * 100
        print(f"     Improvement: {improvement:.1f}%\n")

    print("  ✅ All models saved to models/")
    print("  ✅ model_metadata.json updated\n")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*80)
    print("  ERGO SENSOR -- RETRAINING AMELIORE v2.1")
    print("  38 features  |  meilleurs hyperparametres")
    print("="*80)

    t0 = datetime.now()

    # 1. Load
    df, meta = load_data()

    # 2. Split
    (X_tr, X_te,
     y_reg_tr, y_reg_te,
     y_cls_tr, y_cls_te,
     y_sev_tr, y_sev_te,
     feature_names) = split_data(df)

    # 3. Regressor
    reg_model, reg_metrics, reg_tree_log = train_regressor(
        X_tr, X_te, y_reg_tr, y_reg_te, feature_names
    )

    # 4. Classifier
    n_conditions = len(meta['condition_to_code'])
    cls_model, cls_metrics, cls_tree_log = train_classifier(
        X_tr, X_te, y_cls_tr, y_cls_te, n_conditions, feature_names
    )

    # 5. Severity
    sev_model, sev_metrics, sev_tree_log = train_severity(
        X_tr, X_te, y_sev_tr, y_sev_te, feature_names
    )

    # 6. Anomaly models
    anomaly_results = train_anomaly_models(df, X_tr, X_te, feature_names)

    # 7. Isolation Forest
    iso_model, iso_scaler = train_isolation_forest(X_tr, X_te, feature_names)

    # 8. Update metadata
    update_metadata(meta, reg_metrics, cls_metrics, sev_metrics,
                    feature_names, anomaly_results)

    # Summary
    print_summary(reg_metrics, cls_metrics, sev_metrics,
                  reg_tree_log, cls_tree_log, sev_tree_log)

    elapsed = (datetime.now() - t0).total_seconds()
    print(f"  ⏱️  Total time: {elapsed/60:.1f} min\n")

if __name__ == '__main__':
    main()
