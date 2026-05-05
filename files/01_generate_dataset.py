"""
=============================================================================
ÉTAPE 1 : GÉNÉRATION DU DATASET TMS COMPLET ET ENRICHI - CORPS ENTIER
=============================================================================
Version: 2.0 - Corps complet (12 articulations bilatérales)
Échantillons: 20 000
Corps couverts:
  - Cou (Neck)
  - Tronc (Trunk)
  - Épaule G/D (L/R Shoulder)
  - Coude G/D  (L/R Elbow)
  - Poignet G/D (L/R Wrist)
  - Hanche G/D  (L/R Hip)
  - Genou G/D   (L/R Knee)
=============================================================================
"""

import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import json

warnings.filterwarnings('ignore')

# ============================================================================
# SECTION 1: PARAMÈTRES GLOBAUX
# ============================================================================

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

NB_SAMPLES = 20000
SAMPLING_RATE = 30  # Hz

CONDITION_PRIORITY = [
    "lumbar_disc_hernia_risk",
    "cervical_disc_risk",
    "frozen_shoulder",
    "carpal_tunnel",
    "rotator_cuff_tendinitis",
    "low_back_pain",
    "knee_osteoarthritis",
    "hip_bursitis",
    "elbow_epicondylitis",
    "cervicalgia",
    "shoulder_impingement",
    "de_quervain",
    "tech_neck",
    "postural_kyphosis",
    "wrist_tendinitis",
    "shoulder_bursitis",
    "hip_flexor_strain",
    "knee_tendinitis",
    "elbow_strain",
    "global_msd",
    "muscle_fatigue",
    "normal"
]

THRESHOLDS = {
    # Cou
    'neck_critical': 35, 'neck_high': 25, 'neck_medium': 20,
    # Tronc
    'trunk_critical': 45, 'trunk_high': 30, 'trunk_medium': 20,
    # Épaule
    'shoulder_critical': 80, 'shoulder_high': 65,
    'shoulder_medium': 50, 'shoulder_low': 40,
    # Coude
    'elbow_critical': 120, 'elbow_high': 100, 'elbow_medium': 80,
    # Poignet
    'wrist_high': 30, 'wrist_medium': 20, 'wrist_low': 15,
    # Hanche
    'hip_critical': 60, 'hip_high': 45, 'hip_medium': 30,
    # Genou
    'knee_critical': 90, 'knee_high': 70, 'knee_medium': 50,
    # Risk global
    'risk_critical': 0.85, 'risk_high': 0.65, 'risk_medium': 0.5
}

# ============================================================================
# SECTION 2: GÉNÉRATION DES DONNÉES BRUTES - CORPS ENTIER
# ============================================================================

def generate_raw_data(n_samples=NB_SAMPLES):
    print("[1/6] Génération des données brutes (corps entier)...")

    timestamps = pd.date_range(
        start=datetime(2024, 1, 1),
        periods=n_samples,
        freq=f"{1000/SAMPLING_RATE:.0f}ms"
    )

    data = {
        'Timestamp': timestamps,
        # Cou
        'Neck_Flexion_deg':       np.random.normal(15, 8,  n_samples),
        # Tronc
        'Trunk_Flexion_deg':      np.random.normal(20, 10, n_samples),
        # Épaule droite / gauche
        'R_Shoulder_Flexion_deg': np.random.normal(45, 15, n_samples),
        'L_Shoulder_Flexion_deg': np.random.normal(40, 15, n_samples),
        # Coude droit / gauche  (position neutre ~70°, plage 0-150)
        'R_Elbow_Flexion_deg':    np.random.normal(70, 15, n_samples),
        'L_Elbow_Flexion_deg':    np.random.normal(65, 15, n_samples),
        # Poignet droit / gauche
        'R_Wrist_Deviation_deg':  np.random.normal(10, 5,  n_samples),
        'L_Wrist_Deviation_deg':  np.random.normal(8,  5,  n_samples),
        # Hanche droite / gauche
        'R_Hip_Flexion_deg':      np.random.normal(20, 10, n_samples),
        'L_Hip_Flexion_deg':      np.random.normal(18, 10, n_samples),
        # Genou droit / gauche
        'R_Knee_Flexion_deg':     np.random.normal(15, 10, n_samples),
        'L_Knee_Flexion_deg':     np.random.normal(14, 10, n_samples),
    }

    df = pd.DataFrame(data)

    # --- Patterns réalistes par profil de travail ---

    # 1. Travail bureau prolongé (30% du temps) → cou + tronc + poignets
    desk_idx = np.random.choice(len(df), int(0.30 * len(df)), replace=False)
    df.loc[desk_idx, 'Neck_Flexion_deg']      += np.random.normal(20, 5, len(desk_idx))
    df.loc[desk_idx, 'Trunk_Flexion_deg']     += np.random.normal(15, 5, len(desk_idx))
    df.loc[desk_idx, 'R_Wrist_Deviation_deg'] += np.random.normal(12, 3, len(desk_idx))
    df.loc[desk_idx, 'L_Wrist_Deviation_deg'] += np.random.normal(10, 3, len(desk_idx))

    # 2. Travail en hauteur (15% du temps) → épaules + coudes
    overhead_idx = np.random.choice(len(df), int(0.15 * len(df)), replace=False)
    df.loc[overhead_idx, 'R_Shoulder_Flexion_deg'] += np.random.normal(40, 10, len(overhead_idx))
    df.loc[overhead_idx, 'L_Shoulder_Flexion_deg'] += np.random.normal(35, 10, len(overhead_idx))
    df.loc[overhead_idx, 'R_Elbow_Flexion_deg']    += np.random.normal(20, 8,  len(overhead_idx))

    # 3. Posture de flexion lombaire (20% du temps) → tronc + hanches
    bend_idx = np.random.choice(len(df), int(0.20 * len(df)), replace=False)
    df.loc[bend_idx, 'Trunk_Flexion_deg'] += np.random.normal(30, 8, len(bend_idx))
    df.loc[bend_idx, 'R_Hip_Flexion_deg'] += np.random.normal(25, 8, len(bend_idx))
    df.loc[bend_idx, 'L_Hip_Flexion_deg'] += np.random.normal(22, 8, len(bend_idx))

    # 4. Station debout prolongée (15% du temps) → genoux + hanches
    stand_idx = np.random.choice(len(df), int(0.15 * len(df)), replace=False)
    df.loc[stand_idx, 'R_Knee_Flexion_deg'] += np.random.normal(20, 8, len(stand_idx))
    df.loc[stand_idx, 'L_Knee_Flexion_deg'] += np.random.normal(18, 8, len(stand_idx))
    df.loc[stand_idx, 'R_Hip_Flexion_deg']  += np.random.normal(15, 5, len(stand_idx))

    # 5. Travail poignet intensif (10% du temps)
    wrist_idx = np.random.choice(len(df), int(0.10 * len(df)), replace=False)
    df.loc[wrist_idx, 'R_Wrist_Deviation_deg'] += np.random.normal(22, 4, len(wrist_idx))
    df.loc[wrist_idx, 'L_Wrist_Deviation_deg'] += np.random.normal(18, 4, len(wrist_idx))

    # --- Clamp valeurs réalistes ---
    df['Neck_Flexion_deg']       = df['Neck_Flexion_deg'].clip(0, 90)
    df['Trunk_Flexion_deg']      = df['Trunk_Flexion_deg'].clip(0, 90)
    df['R_Shoulder_Flexion_deg'] = df['R_Shoulder_Flexion_deg'].clip(0, 180)
    df['L_Shoulder_Flexion_deg'] = df['L_Shoulder_Flexion_deg'].clip(0, 180)
    df['R_Elbow_Flexion_deg']    = df['R_Elbow_Flexion_deg'].clip(0, 150)
    df['L_Elbow_Flexion_deg']    = df['L_Elbow_Flexion_deg'].clip(0, 150)
    df['R_Wrist_Deviation_deg']  = df['R_Wrist_Deviation_deg'].clip(0, 45)
    df['L_Wrist_Deviation_deg']  = df['L_Wrist_Deviation_deg'].clip(0, 45)
    df['R_Hip_Flexion_deg']      = df['R_Hip_Flexion_deg'].clip(0, 120)
    df['L_Hip_Flexion_deg']      = df['L_Hip_Flexion_deg'].clip(0, 120)
    df['R_Knee_Flexion_deg']     = df['R_Knee_Flexion_deg'].clip(0, 135)
    df['L_Knee_Flexion_deg']     = df['L_Knee_Flexion_deg'].clip(0, 135)

    df = df.ffill().bfill()
    return df

# ============================================================================
# SECTION 3: FEATURES BIOMÉCANIQUES
# ============================================================================

def compute_biomechanical_features(df):
    print("[2/6] Calcul des features biomécaniques...")

    df['neck']       = abs(df['Neck_Flexion_deg'])
    df['trunk']      = abs(df['Trunk_Flexion_deg'])

    # Épaules : max gauche/droite + valeur individuelle
    df['r_shoulder'] = abs(df['R_Shoulder_Flexion_deg'])
    df['l_shoulder'] = abs(df['L_Shoulder_Flexion_deg'])
    df['shoulder']   = df[['r_shoulder', 'l_shoulder']].max(axis=1)

    # Coudes
    df['r_elbow']    = abs(df['R_Elbow_Flexion_deg'])
    df['l_elbow']    = abs(df['L_Elbow_Flexion_deg'])
    df['elbow']      = df[['r_elbow', 'l_elbow']].max(axis=1)

    # Poignets
    df['r_wrist']    = abs(df['R_Wrist_Deviation_deg'])
    df['l_wrist']    = abs(df['L_Wrist_Deviation_deg'])
    df['wrist']      = df[['r_wrist', 'l_wrist']].max(axis=1)

    # Hanches
    df['r_hip']      = abs(df['R_Hip_Flexion_deg'])
    df['l_hip']      = abs(df['L_Hip_Flexion_deg'])
    df['hip']        = df[['r_hip', 'l_hip']].max(axis=1)

    # Genoux
    df['r_knee']     = abs(df['R_Knee_Flexion_deg'])
    df['l_knee']     = abs(df['L_Knee_Flexion_deg'])
    df['knee']       = df[['r_knee', 'l_knee']].max(axis=1)

    return df

# ============================================================================
# SECTION 4: FEATURES TEMPORELLES
# ============================================================================

def compute_temporal_features(df, window_size=30):
    print("[3/6] Calcul des features temporelles...")

    joints = ['neck', 'trunk', 'shoulder', 'elbow', 'wrist', 'hip', 'knee']

    for j in joints:
        df[f'{j}_vel']      = df[j].diff().fillna(0).abs()
        df[f'{j}_duration'] = (df[j] > THRESHOLDS.get(f'{j}_medium', 20)).astype(int).rolling(window_size, min_periods=1).sum()
        df[f'{j}_freq']     = (df[f'{j}_vel'] > df[f'{j}_vel'].quantile(0.75)).astype(int).rolling(window_size, min_periods=1).sum()

    return df

# ============================================================================
# SECTION 5: RISK SCORE GLOBAL
# ============================================================================

def compute_risk_score(df):
    print("[4/6] Calcul du risk_score global...")

    # Pondérations ajustées : équilibre entre tous les membres
    # Le risk_score brut peut dépasser 1 → pondération totale = 1.0
    df['risk_score_raw'] = (
        0.22 * (df['neck']     / 90)   +
        0.22 * (df['trunk']    / 90)   +
        0.14 * (df['shoulder'] / 180)  +
        0.10 * (df['elbow']    / 150)  +
        0.12 * (df['wrist']    / 45)   +
        0.10 * (df['hip']      / 120)  +
        0.10 * (df['knee']     / 135)
    )
    # Amplification pour avoir une distribution de sévérité réaliste
    # (sinon tout reste en 'low' avec des angles faibles)
    df['risk_score_raw'] = (df['risk_score_raw'] * 1.8).clip(0, 1)

    return df

# ============================================================================
# SECTION 6: CONDITIONS MULTI-LABEL - CORPS ENTIER
# ============================================================================

def generate_conditions(row, thresholds=THRESHOLDS):
    conditions = []

    # Cou
    if row['neck'] > thresholds['neck_critical']:
        conditions.append("cervical_disc_risk")
    if row['neck'] > thresholds['neck_high']:
        conditions.append("cervicalgia")
    if row['neck'] > thresholds['neck_medium']:
        conditions.append("tech_neck")

    # Tronc
    if row['trunk'] > thresholds['trunk_critical']:
        conditions.append("lumbar_disc_hernia_risk")
    if row['trunk'] > thresholds['trunk_high']:
        conditions.append("low_back_pain")
    if row['trunk'] > thresholds['trunk_medium']:
        conditions.append("postural_kyphosis")

    # Épaule
    if row['shoulder'] > thresholds['shoulder_critical']:
        conditions.append("frozen_shoulder")
    if row['shoulder'] > thresholds['shoulder_high']:
        conditions.append("rotator_cuff_tendinitis")
    if row['shoulder'] > thresholds['shoulder_medium']:
        conditions.append("shoulder_impingement")
    if row['shoulder'] > thresholds['shoulder_low']:
        conditions.append("shoulder_bursitis")

    # Coude
    if row['elbow'] > thresholds['elbow_critical']:
        conditions.append("elbow_epicondylitis")
    if row['elbow'] > thresholds['elbow_high']:
        conditions.append("elbow_strain")

    # Poignet
    if row['wrist'] > thresholds['wrist_high']:
        conditions.append("carpal_tunnel")
    if row['wrist'] > thresholds['wrist_medium']:
        conditions.append("de_quervain")
    if row['wrist'] > thresholds['wrist_low']:
        conditions.append("wrist_tendinitis")

    # Hanche
    if row['hip'] > thresholds['hip_critical']:
        conditions.append("hip_bursitis")
    if row['hip'] > thresholds['hip_high']:
        conditions.append("hip_flexor_strain")

    # Genou
    if row['knee'] > thresholds['knee_critical']:
        conditions.append("knee_osteoarthritis")
    if row['knee'] > thresholds['knee_high']:
        conditions.append("knee_tendinitis")

    # Global
    if row['risk_score_raw'] > thresholds['risk_critical']:
        conditions.append("global_msd")
    elif row['risk_score_raw'] > thresholds['risk_high']:
        conditions.append("muscle_fatigue")

    if len(conditions) == 0:
        conditions.append("normal")

    return sorted(list(set(conditions)))


def compute_conditions(df):
    print("[5/6] Génération des conditions multi-label...")
    df['conditions'] = df.apply(lambda row: generate_conditions(row), axis=1)
    return df

# ============================================================================
# SECTION 7: FEATURES DÉRIVÉES
# ============================================================================

def select_main_condition(conditions, priority=CONDITION_PRIORITY):
    for p in priority:
        if p in conditions:
            return p
    return conditions[0]


def compute_derived_features(df):
    print("[5/6] Calcul des features dérivées...")

    df['main_condition'] = df['conditions'].apply(select_main_condition)

    def severity_level(score):
        if score > THRESHOLDS['risk_critical']:
            return "high"
        elif score > THRESHOLDS['risk_high']:
            return "medium"
        else:
            return "low"

    df['severity'] = df['risk_score_raw'].apply(severity_level)

    def get_location(row):
        stress = {
            'neck':     row['neck'],
            'trunk':    row['trunk'],
            'shoulder': row['shoulder'],
            'elbow':    row['elbow'],
            'wrist':    row['wrist'],
            'hip':      row['hip'],
            'knee':     row['knee'],
        }
        return max(stress, key=stress.get)

    df['location'] = df.apply(get_location, axis=1)

    threshold_anomaly = df['risk_score_raw'].quantile(0.95)
    df['anomaly'] = (df['risk_score_raw'] > threshold_anomaly).astype(int)

    return df

# ============================================================================
# SECTION 8: NORMALISATION
# ============================================================================

def normalize_risk_score(df):
    print("[5/6] Normalisation du risk_score...")
    scaler = MinMaxScaler()
    df['risk_score'] = scaler.fit_transform(df[['risk_score_raw']])
    return df, scaler

# ============================================================================
# SECTION 9: ENCODAGE CATÉGORIQUE
# ============================================================================

def encode_categorical_features(df):
    print("[5/6] Encodage des features catégoriques...")

    unique_conditions = sorted(df['main_condition'].unique())
    condition_to_code = {c: i for i, c in enumerate(unique_conditions)}
    df['condition_code'] = df['main_condition'].map(condition_to_code)

    location_to_code = {loc: i for i, loc in enumerate(sorted(df['location'].unique()))}
    df['location_code'] = df['location'].map(location_to_code)

    severity_to_code = {'low': 0, 'medium': 1, 'high': 2}
    df['severity_code'] = df['severity'].map(severity_to_code)

    mappings = {
        'condition_to_code': condition_to_code,
        'location_to_code': location_to_code,
        'severity_to_code': severity_to_code
    }
    return df, mappings

# ============================================================================
# SECTION 10: FEATURES ML
# ============================================================================

def define_ml_features():
    basic_features = [
        # Angles agrégés
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
        # Fréquences de mouvement
        'neck_freq', 'trunk_freq', 'shoulder_freq', 'elbow_freq',
        'wrist_freq', 'hip_freq', 'knee_freq',
    ]
    return {
        'basic': basic_features,
        'target_regression':     'risk_score',
        'target_classification': 'condition_code',
        'target_severity':       'severity_code',
        'timestamp':             'Timestamp'
    }

# ============================================================================
# SECTION 11: PIPELINE COMPLET
# ============================================================================

def generate_complete_dataset(n_samples=NB_SAMPLES, output_path='dataset_TMS_enriched.csv'):
    print("\n" + "="*80)
    print("GÉNÉRATION DU DATASET TMS - CORPS ENTIER")
    print("="*80 + "\n")

    df = generate_raw_data(n_samples)
    df = compute_biomechanical_features(df)
    df = compute_temporal_features(df)
    df = compute_risk_score(df)
    df = compute_conditions(df)
    df = compute_derived_features(df)
    df, scaler = normalize_risk_score(df)
    df, mappings = encode_categorical_features(df)

    print("\n[6/6] Sauvegarde du dataset...")
    df.to_csv(output_path, index=False)

    with open('condition_mappings.json', 'w') as f:
        json.dump(mappings, f, indent=2)

    import pickle
    with open('risk_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Statistiques
    features = define_ml_features()
    print("\n" + "="*80)
    print("STATISTIQUES DU DATASET GENERE")
    print("="*80)
    print(f"[OK] Nombre de samples    : {len(df)}")
    print(f"[OK] Nombre de features ML: {len(features['basic'])}")
    print(f"[OK] Articulations: neck, trunk, shoulder(G/D), elbow(G/D), wrist(G/D), hip(G/D), knee(G/D)")
    print(f"\n[INFO] Conditions uniques: {df['main_condition'].nunique()}")
    print(df['main_condition'].value_counts().to_string())
    print(f"\n[INFO] Severite distribution:")
    print(df['severity'].value_counts().to_string())
    print(f"\n[INFO] Localisation distribution:")
    print(df['location'].value_counts().to_string())
    print(f"\n[WARN] Anomalies: {df['anomaly'].sum()} ({100*df['anomaly'].mean():.1f}%)")
    print(f"\n[INFO] Risk score -- Mean: {df['risk_score'].mean():.4f} | Std: {df['risk_score'].std():.4f}")
    print("="*80)

    return df, mappings, scaler


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    df, mappings, scaler = generate_complete_dataset()

    print("\n[INFO] APERCU (5 premieres lignes):\n")
    print(df[[
        'Timestamp', 'neck', 'trunk', 'shoulder', 'elbow',
        'wrist', 'hip', 'knee', 'risk_score', 'severity',
        'main_condition', 'location', 'anomaly'
    ]].head())

    print("\n[OK] Dataset genere avec succes!")
    print("     dataset_TMS_enriched.csv")
    print("     condition_mappings.json")
    print("     risk_scaler.pkl")
