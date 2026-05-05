"""
=============================================================================
ETAPE 2 : ENTRAINEMENT DES MODELES ML - LIGHTGBM ONLY (v2.0)
=============================================================================
Corps entier - 12 articulations bilatérales - 20 000 echantillons
Sans TensorFlow (incompatible). Modeles entraine:
   1. LightGBM Regression  -> risk_score       (lgb_regressor.txt)
   2. LightGBM Classifier  -> main_condition    (lgb_classifier.txt)
   3. LightGBM Severite    -> severity_code     (lgb_severity.txt)
   4. IsolationForest      -> anomaly global    (isolation_forest.pkl)
   5. LightGBM x5 anomaly  -> par articulation  (lgbm_anomaly_*.txt)
   + model_metadata.json compatible avec ai_engine.py
=============================================================================
"""

import os
import pandas as pd
import numpy as np
import json
import pickle
import warnings
from datetime import datetime

import lightgbm as lgb
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ============================================================================
# SECTION 1: PARAMETRES
# ============================================================================

MODELS_DIR = 'models'
PLOTS_DIR  = 'plots'
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

# Features completes corps entier
FEATURES_BASIC = [
    # Angles agrégés (max G/D)
    'neck', 'trunk', 'shoulder', 'elbow', 'wrist', 'hip', 'knee',
    # Angles bilatéraux
    'r_shoulder', 'l_shoulder',
    'r_elbow',    'l_elbow',
    'r_wrist',    'l_wrist',
    'r_hip',      'l_hip',
    'r_knee',     'l_knee',
    # Vélocités
    'neck_vel', 'trunk_vel', 'shoulder_vel', 'elbow_vel',
    'wrist_vel', 'hip_vel', 'knee_vel',
    # Durées en mauvaise posture
    'neck_duration', 'trunk_duration', 'shoulder_duration', 'elbow_duration',
    'wrist_duration', 'hip_duration', 'knee_duration',
    # Fréquences
    'neck_freq', 'trunk_freq', 'shoulder_freq', 'elbow_freq',
    'wrist_freq', 'hip_freq', 'knee_freq',
]

# Colonnes d'anomalie (pour les 5 detecteurs specifiques)
ANOMALY_COLS = [
    'anomaly_neck_hyperflex',
    'anomaly_shoulder_overext',
    'anomaly_wrist_strain',
    'anomaly_trunk_torsion',
    'anomaly_elbow_hyperext',
]
ANOMALY_NAMES = [
    'Neck Hyperflexion',
    'Shoulder Overextension',
    'Wrist Strain',
    'Trunk Torsion',
    'Elbow Hyperextension',
]

# ============================================================================
# SECTION 2: CHARGEMENT ET PREPROCESSING
# ============================================================================

def load_dataset(path='dataset_TMS_enriched.csv'):
    print("[1/7] Chargement du dataset...")
    df = pd.read_csv(path)
    with open('condition_mappings.json', 'r') as f:
        mappings = json.load(f)
    print(f"  [OK] {len(df)} samples | {len(df.columns)} colonnes")
    return df, mappings


def split_temporal(df, features, test_size=0.2):
    print("[2/7] Split temporel train/test...")
    n = len(df)
    split = int(n * (1 - test_size))

    X_train = df[features].iloc[:split].copy()
    X_test  = df[features].iloc[split:].copy()

    y_train_reg = df['risk_score'].iloc[:split]
    y_test_reg  = df['risk_score'].iloc[split:]

    y_train_cls = df['condition_code'].iloc[:split]
    y_test_cls  = df['condition_code'].iloc[split:]

    y_train_sev = df['severity_code'].iloc[:split]
    y_test_sev  = df['severity_code'].iloc[split:]

    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
    return (X_train, X_test,
            y_train_reg, y_test_reg,
            y_train_cls, y_test_cls,
            y_train_sev, y_test_sev)


def scale_features(X_train, X_test, features):
    print("[3/7] Normalisation StandardScaler...")
    scaler = StandardScaler()
    Xs_train = pd.DataFrame(scaler.fit_transform(X_train), columns=features, index=X_train.index)
    Xs_test  = pd.DataFrame(scaler.transform(X_test),      columns=features, index=X_test.index)
    return Xs_train, Xs_test, scaler


# ============================================================================
# SECTION 3: MODELE 1 - LIGHTGBM REGRESSION (risk_score)
# ============================================================================

def train_lgb_regressor(X_train, X_test, y_train, y_test):
    print("\n" + "="*70)
    print("MODELE 1: LightGBM Regression (risk_score)")
    print("="*70)

    train_data = lgb.Dataset(X_train, label=y_train)
    test_data  = lgb.Dataset(X_test,  label=y_test, reference=train_data)

    params = {
        'objective':        'regression',
        'metric':           'rmse',
        'learning_rate':    0.01,
        'num_leaves':       63,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq':     5,
        'min_child_samples': 20,
        'verbose':          -1,
        'seed':             42,
    }

    model = lgb.train(
        params, train_data,
        num_boost_round=800,
        valid_sets=[test_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)]
    )

    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    print(f"\n  [OK] MAE={mae:.6f} | RMSE={rmse:.6f} | R2={r2:.6f}")

    path = f'{MODELS_DIR}/lgb_regressor.txt'
    model.save_model(path)
    print(f"  [SAVE] {path}")
    return model, {'mae': mae, 'rmse': rmse, 'r2': r2}


# ============================================================================
# SECTION 4: MODELE 2 - LIGHTGBM CLASSIFICATION (main_condition)
# ============================================================================

def train_lgb_classifier(X_train, X_test, y_train, y_test, n_classes):
    print("\n" + "="*70)
    print("MODELE 2: LightGBM Classification (condition)")
    print("="*70)

    model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=n_classes,
        learning_rate=0.01,
        num_leaves=63,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=5,
        min_child_samples=20,
        verbose=-1,
        random_state=42,
        n_estimators=800,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50)]
    )

    y_pred = model.predict(X_test)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1   = f1_score(y_test, y_pred, average='macro', zero_division=0)

    print(f"\n  [OK] Accuracy={acc:.4f} | Precision={prec:.4f} | Recall={rec:.4f} | F1={f1:.4f}")

    path = f'{MODELS_DIR}/lgb_classifier.txt'
    model.booster_.save_model(path)
    print(f"  [SAVE] {path}")
    return model, {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}


# ============================================================================
# SECTION 5: MODELE 3 - LIGHTGBM SEVERITE
# ============================================================================

def train_lgb_severity(X_train, X_test, y_train, y_test):
    print("\n" + "="*70)
    print("MODELE 3: LightGBM Classification (severity)")
    print("="*70)

    model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=3,
        learning_rate=0.01,
        num_leaves=31,
        feature_fraction=0.9,
        verbose=-1,
        random_state=42,
        n_estimators=500,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50)]
    )

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='macro', zero_division=0)
    print(f"\n  [OK] Accuracy={acc:.4f} | F1={f1:.4f}")

    path = f'{MODELS_DIR}/lgb_severity.txt'
    model.booster_.save_model(path)
    print(f"  [SAVE] {path}")
    return model, {'accuracy': acc, 'f1': f1}


# ============================================================================
# SECTION 6: MODELE 4 - ISOLATION FOREST (anomalie globale)
# ============================================================================

def train_isolation_forest(X_train, X_test, features):
    print("\n" + "="*70)
    print("MODELE 4: IsolationForest (anomalie globale)")
    print("="*70)

    # IsolationForest s'entraîne sur X brut (pas normalisé de la même façon)
    scaler_if = StandardScaler()
    Xif_train = scaler_if.fit_transform(X_train)
    Xif_test  = scaler_if.transform(X_test)

    iso = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42,
        n_jobs=-1
    )
    iso.fit(Xif_train)

    scores_train = iso.decision_function(Xif_train)
    scores_test  = iso.decision_function(Xif_test)
    n_anom = (iso.predict(Xif_test) == -1).sum()
    print(f"\n  [OK] Anomalies detectees dans test: {n_anom} ({100*n_anom/len(Xif_test):.1f}%)")

    joblib.dump(iso,       f'{MODELS_DIR}/isolation_forest.pkl')
    joblib.dump(scaler_if, f'{MODELS_DIR}/scaler_if.pkl')
    print(f"  [SAVE] isolation_forest.pkl + scaler_if.pkl")
    return iso, scaler_if


# ============================================================================
# SECTION 7: MODELES 5-9 - ANOMALIES PAR ARTICULATION
# ============================================================================

def generate_anomaly_labels(df_train, df_test):
    """
    Cree des labels binaires d'anomalie pour chaque articulation.
    Approche: top 5% de chaque angle = anomalie.
    """
    thresholds = {
        'anomaly_neck_hyperflex':   df_train['neck'].quantile(0.95),
        'anomaly_shoulder_overext': df_train['shoulder'].quantile(0.95),
        'anomaly_wrist_strain':     df_train['wrist'].quantile(0.95),
        'anomaly_trunk_torsion':    df_train['trunk'].quantile(0.95),
        'anomaly_elbow_hyperext':   df_train['elbow'].quantile(0.95),
    }
    sources = {
        'anomaly_neck_hyperflex':   'neck',
        'anomaly_shoulder_overext': 'shoulder',
        'anomaly_wrist_strain':     'wrist',
        'anomaly_trunk_torsion':    'trunk',
        'anomaly_elbow_hyperext':   'elbow',
    }
    labels = {}
    for col, src in sources.items():
        thr = thresholds[col]
        labels[f'{col}_train'] = (df_train[src] > thr).astype(int).values
        labels[f'{col}_test']  = (df_test[src]  > thr).astype(int).values
    return labels, thresholds


def train_anomaly_models(X_train_raw, X_test_raw, df_train, df_test, features):
    print("\n" + "="*70)
    print("MODELES 5-9: LightGBM Anomalie par articulation")
    print("="*70)

    labels, thresholds = generate_anomaly_labels(df_train, df_test)
    anomaly_models = {}

    for col in ANOMALY_COLS:
        y_tr = labels[f'{col}_train']
        y_te = labels[f'{col}_test']

        # Vérifier qu'il y a des positifs
        if y_tr.sum() < 5:
            print(f"  [SKIP] {col} — pas assez de positifs ({y_tr.sum()})")
            continue

        model = lgb.LGBMClassifier(
            objective='binary',
            learning_rate=0.05,
            num_leaves=31,
            n_estimators=300,
            verbose=-1,
            random_state=42,
            is_unbalance=True,
        )
        model.fit(X_train_raw, y_tr)

        y_pred = model.predict(X_test_raw)
        acc = accuracy_score(y_te, y_pred)
        f1  = f1_score(y_te, y_pred, zero_division=0)
        print(f"  [OK] {col}: Accuracy={acc:.4f} | F1={f1:.4f} | Positifs train={y_tr.sum()}")

        path = f'{MODELS_DIR}/lgbm_{col}.txt'
        model.booster_.save_model(path)
        anomaly_models[col] = model

    print(f"\n  [SAVE] {len(anomaly_models)}/5 modeles anomalie sauvegardes")
    return anomaly_models, thresholds


# ============================================================================
# SECTION 8: SHAP EXPLAINABILITY
# ============================================================================

def compute_shap(model_lgb_reg, X_test_scaled, features):
    print("\n[6/7] Calcul SHAP...")
    explainer   = shap.TreeExplainer(model_lgb_reg)
    sample      = X_test_scaled.sample(min(200, len(X_test_scaled)), random_state=42)
    shap_values = explainer.shap_values(sample)

    plt.figure(figsize=(14, 8))
    shap.summary_plot(shap_values, sample, plot_type='bar', show=False)
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/shap_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVE] {PLOTS_DIR}/shap_summary.png")
    return explainer, shap_values


# ============================================================================
# SECTION 9: METADATA JSON (compatible ai_engine.py)
# ============================================================================

def save_metadata(mappings, features, scaler_if, thresholds, results):
    print("\n[7/7] Sauvegarde model_metadata.json...")

    # Features pour IsolationForest (toutes les features de base)
    if_features = features

    meta = {
        "version":       "2.0",
        "created":       datetime.now().isoformat(),
        "n_samples":     20000,
        "n_features":    len(features),
        "feature_cols":  features,
        "if_features":   if_features,
        "seq_len":       60,
        "condition_to_code": mappings['condition_to_code'],
        "severity_to_code":  mappings['severity_to_code'],
        "location_to_code":  mappings['location_to_code'],
        "anomaly_cols":  ANOMALY_COLS,
        "anomaly_names": ANOMALY_NAMES,
        "body_parts": [
            "neck", "trunk",
            "r_shoulder", "l_shoulder",
            "r_elbow",    "l_elbow",
            "r_wrist",    "l_wrist",
            "r_hip",      "l_hip",
            "r_knee",     "l_knee",
        ],
        "metrics": results,
        "anomaly_thresholds": {k: float(v) for k, v in thresholds.items()},
    }

    path = f'{MODELS_DIR}/model_metadata.json'
    with open(path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  [SAVE] {path}")
    return meta


# ============================================================================
# SECTION 10: RAPPORT FINAL
# ============================================================================

def print_report(results):
    print("\n" + "="*70)
    print("RAPPORT FINAL")
    print("="*70)
    for name, metrics in results.items():
        print(f"\n  {name}:")
        for k, v in metrics.items():
            print(f"    {k}: {v:.6f}")
    print("\n" + "="*70)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

if __name__ == "__main__":

    print("\n" + "="*70)
    print("PIPELINE ENTRAINEMENT COMPLET - CORPS ENTIER")
    print("="*70 + "\n")

    # 1. Charger
    df, mappings = load_dataset('dataset_TMS_enriched.csv')

    # 2. Split
    (X_train, X_test,
     y_tr_reg, y_te_reg,
     y_tr_cls, y_te_cls,
     y_tr_sev, y_te_sev) = split_temporal(df, FEATURES_BASIC)

    # Garder refs brutes pour anomalie
    df_train = df.iloc[:len(X_train)]
    df_test  = df.iloc[len(X_train):]

    # 3. Normaliser
    Xs_train, Xs_test, feat_scaler = scale_features(X_train, X_test, FEATURES_BASIC)

    results = {}

    # 4. Modele 1 : regression
    print("\n[4/7] Entrainement des modeles...")
    lgb_reg, res_reg = train_lgb_regressor(Xs_train, Xs_test, y_tr_reg, y_te_reg)
    results['LightGBM_Regression'] = res_reg

    # 5. Modele 2 : classification condition
    n_classes = len(mappings['condition_to_code'])
    lgb_cls, res_cls = train_lgb_classifier(Xs_train, Xs_test, y_tr_cls, y_te_cls, n_classes)
    results['LightGBM_Classifier'] = res_cls

    # 6. Modele 3 : severite
    lgb_sev, res_sev = train_lgb_severity(Xs_train, Xs_test, y_tr_sev, y_te_sev)
    results['LightGBM_Severity'] = res_sev

    # 7. Modele 4 : IsolationForest
    iso, scaler_if = train_isolation_forest(X_train, X_test, FEATURES_BASIC)
    results['IsolationForest'] = {'trained': 1.0}

    # 8. Modeles 5-9 : anomalie par articulation
    anomaly_models, anom_thresholds = train_anomaly_models(
        Xs_train, Xs_test, df_train, df_test, FEATURES_BASIC
    )
    results['Anomaly_Models'] = {'count': float(len(anomaly_models))}

    # 9. SHAP
    explainer, shap_vals = compute_shap(lgb_reg, Xs_test, FEATURES_BASIC)

    # 10. Sauvegarder preprocessors
    joblib.dump(feat_scaler, f'{MODELS_DIR}/feature_scaler.pkl')
    print(f"  [SAVE] {MODELS_DIR}/feature_scaler.pkl")

    # 11. Metadata JSON
    meta = save_metadata(mappings, FEATURES_BASIC, scaler_if, anom_thresholds, results)

    # 12. Rapport
    print_report(results)

    print("\n[OK] Tous les modeles entraines et sauvegardes dans models/")
    print("     Relancez app.py pour utiliser les nouveaux modeles.\n")
