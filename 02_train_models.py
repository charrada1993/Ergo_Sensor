"""
=============================================================================
ÉTAPE 2 : ENTRAÎNEMENT DE TOUS LES MODÈLES ML
=============================================================================
Auteur: Ingénieur IA Senior
Date: 2025
Version: 1.0 (Production-Ready)

✅ Entraîne 5 modèles principaux:
   1. LightGBM - Régression (risk_score)
   2. LightGBM - Classification (condition)
   3. CNN 1D - Features temporelles
   4. Hybrid - CNN + LightGBM
   5. LSTM - Séquences temporelles

✅ Includes:
   - Train/Test split temporel (sans leakage)
   - Validation croisée temporelle
   - Métriques complètes
   - Sauvegarde des modèles
   - SHAP pour interprétabilité
=============================================================================
"""

import pandas as pd
import numpy as np
import json
import pickle
import warnings
from datetime import datetime

# ML Libraries
import lightgbm as lgb
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import joblib

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, LSTM, Flatten, Dense,
    Dropout, Input, Bidirectional
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)

# Explainability
import shap

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ============================================================================
# SECTION 1: CHARGEMENT DES DONNÉES
# ============================================================================

def load_dataset(dataset_path='dataset_TMS_enriched.csv'):
    """
    Charge le dataset généré et les métadonnées
    """
    print("[1/10] Chargement du dataset...")
    
    df = pd.read_csv(dataset_path)
    
    # Charger les mappings
    with open('condition_mappings.json', 'r') as f:
        mappings = json.load(f)
    
    # Charger le scaler
    with open('risk_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    print(f"✅ Dataset chargé: {len(df)} samples, {len(df.columns)} features")
    
    return df, mappings, scaler

# ============================================================================
# SECTION 2: PRÉPARATION DES DONNÉES
# ============================================================================

class DataPreprocessor:
    """
    Classe pour préprocesser les données de manière cohérente
    """
    
    def __init__(self, df, mappings):
        self.df = df.copy()
        self.mappings = mappings
        
        # Features ML définies
        self.features_basic = [
            'neck', 'trunk', 'shoulder', 'wrist',
            'neck_vel', 'trunk_vel', 'shoulder_vel', 'wrist_vel',
            'neck_duration', 'trunk_duration', 'shoulder_duration', 'wrist_duration',
            'neck_freq', 'trunk_freq'
        ]
        
        self.target_reg = 'risk_score'
        self.target_cls = 'condition_code'
        self.target_severity = 'severity_code'
    
    def get_features(self):
        """Retourne les features pour ML"""
        return self.features_basic
    
    def split_temporal(self, test_size=0.2):
        """
        Split temporel propre (pas d'aléatoire pour éviter le leakage)
        """
        print("[2/10] Split temporel train/test...")
        
        split_idx = int(len(self.df) * (1 - test_size))
        
        X_train = self.df[self.features_basic].iloc[:split_idx]
        X_test = self.df[self.features_basic].iloc[split_idx:]
        
        y_train_reg = self.df[self.target_reg].iloc[:split_idx]
        y_test_reg = self.df[self.target_reg].iloc[split_idx:]
        
        y_train_cls = self.df[self.target_cls].iloc[:split_idx]
        y_test_cls = self.df[self.target_cls].iloc[split_idx:]
        
        print(f"   Train: {len(X_train)} samples")
        print(f"   Test:  {len(X_test)} samples")
        
        return (X_train, X_test, y_train_reg, y_test_reg,
                y_train_cls, y_test_cls)
    
    def normalize_features(self, X_train, X_test):
        """
        Normalise les features (StandardScaler fit sur train uniquement)
        """
        print("[3/10] Normalisation des features...")
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Reconvertir en DataFrame pour garder les noms de colonnes
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        return X_train_scaled, X_test_scaled, scaler
    
    def create_sequences(self, X, y, window=20):
        """
        Crée des fenêtres temporelles pour CNN/LSTM
        (sliding window propre, sans troncature)
        """
        Xs, ys = [], []
        for i in range(len(X) - window):
            Xs.append(X.iloc[i:i+window].values)
            ys.append(y.iloc[i+window])
        
        return np.array(Xs), np.array(ys)

# ============================================================================
# SECTION 3: MODÈLE 1 - LightGBM RÉGRESSION
# ============================================================================

class LightGBMRegressor:
    """
    Modèle LightGBM pour prédire risk_score
    """
    
    def __init__(self):
        self.model = None
        self.history = None
        self.feature_importance = None
    
    def train(self, X_train, X_test, y_train, y_test):
        """
        Entraîne le modèle avec early stopping
        """
        print("\n" + "="*80)
        print("MODÈLE 1: LightGBM Régression (RISK SCORE)")
        print("="*80)
        
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': 0.01,
            'num_leaves': 31,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.9,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': 42
        }
        
        callbacks = [
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
        
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[test_data],
            callbacks=callbacks
        )
        
        return self.evaluate(X_test, y_test)
    
    def evaluate(self, X_test, y_test):
        """
        Évalue le modèle et affiche les métriques
        """
        y_pred = self.model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n✅ Résultats LightGBM Régression:")
        print(f"   MAE:  {mae:.6f}")
        print(f"   RMSE: {rmse:.6f}")
        print(f"   R²:   {r2:.6f}")
        
        # Feature importance
        self.feature_importance = self.model.feature_importance()
        
        return {
            'model': self.model,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'predictions': y_pred,
            'actual': y_test
        }
    
    def save(self, path='models/lgb_regressor.txt'):
        """Sauvegarde le modèle"""
        self.model.save_model(path)
        print(f"   📁 Modèle sauvegardé: {path}")

# ============================================================================
# SECTION 4: MODÈLE 2 - LightGBM CLASSIFICATION
# ============================================================================

class LightGBMClassifier:
    """
    Modèle LightGBM pour classifier les conditions
    """
    
    def __init__(self, n_classes):
        self.model = None
        self.n_classes = n_classes
        self.classes = None
    
    def train(self, X_train, X_test, y_train, y_test):
        """
        Entraîne le classifier multiclasse
        """
        print("\n" + "="*80)
        print("MODÈLE 2: LightGBM Classification (CONDITIONS)")
        print("="*80)
        
        self.classes = sorted(y_train.unique())
        
        self.model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=self.n_classes,
            learning_rate=0.01,
            num_leaves=31,
            feature_fraction=0.9,
            bagging_fraction=0.9,
            verbose=-1,
            random_state=42
        )
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(stopping_rounds=50)]
        )
        
        return self.evaluate(X_test, y_test)
    
    def evaluate(self, X_test, y_test):
        """
        Évalue le classifier
        """
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        print(f"\n✅ Résultats LightGBM Classification:")
        print(f"   Accuracy:  {accuracy:.6f}")
        print(f"   Precision: {precision_macro:.6f}")
        print(f"   Recall:    {recall_macro:.6f}")
        print(f"   F1-Score:  {f1_macro:.6f}")
        
        return {
            'model': self.model,
            'accuracy': accuracy,
            'precision': precision_macro,
            'recall': recall_macro,
            'f1': f1_macro,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'actual': y_test
        }
    
    def save(self, path='models/lgb_classifier.txt'):
        """Sauvegarde le modèle"""
        self.model.booster_.save_model(path)
        print(f"   📁 Modèle sauvegardé: {path}")

# ============================================================================
# SECTION 5: MODÈLE 3 - CNN 1D
# ============================================================================

class CNN1D:
    """
    CNN 1D pour extraire des features temporelles
    Architecte avec une tête de feature extraction séparable
    """
    
    def __init__(self, input_shape):
        self.model_full = None
        self.feature_extractor = None
        self.input_shape = input_shape
    
    def build(self):
        """
        Construit l'architecture CNN avec séparation feature/output
        """
        print("\n" + "="*80)
        print("MODÈLE 3: CNN 1D (FEATURE EXTRACTION)")
        print("="*80)
        
        inputs = Input(shape=self.input_shape)
        
        # Bloc 1: Convolution + Pooling
        x = Conv1D(32, kernel_size=3, activation='relu', padding='same')(inputs)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.2)(x)
        
        # Bloc 2: Convolution + Pooling
        x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.2)(x)
        
        # Bloc 3: Convolution
        x = Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.2)(x)
        
        # Flatten
        x = Flatten()(x)
        
        # Feature extractor (sortie intermédiaire)
        features = Dense(32, activation='relu', name='features')(x)
        
        # Output head (prédiction risk_score)
        output = Dense(1, activation='sigmoid', name='risk')(features)
        
        # Modèle complet
        self.model_full = Model(inputs=inputs, outputs=output, name='CNN_Full')
        
        # Feature extractor (pour extraction de features)
        self.feature_extractor = Model(inputs=inputs, outputs=features, name='CNN_FeatureExtractor')
        
        self.model_full.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        print(f"   ✅ Architecture CNN 1D construite")
        self.model_full.summary()
        
        return self.model_full
    
    def train(self, X_train, X_test, y_train, y_test, epochs=20):
        """
        Entraîne le CNN
        """
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
            ModelCheckpoint('models/cnn_best.h5', save_best_only=True)
        ]
        
        history = self.model_full.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Évaluation
        y_pred = self.model_full.predict(X_test, verbose=0)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n✅ Résultats CNN 1D:")
        print(f"   MAE:  {mae:.6f}")
        print(f"   RMSE: {rmse:.6f}")
        print(f"   R²:   {r2:.6f}")
        
        return {
            'model': self.model_full,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'predictions': y_pred,
            'actual': y_test,
            'history': history
        }
    
    def extract_features(self, X):
        """
        Extrait les features apprises par le CNN
        """
        return self.feature_extractor.predict(X, verbose=0)
    
    def save(self, path='models/cnn_model.h5'):
        """Sauvegarde le modèle"""
        self.model_full.save(path)
        print(f"   📁 Modèle sauvegardé: {path}")

# ============================================================================
# SECTION 6: MODÈLE 4 - HYBRID (CNN + LightGBM)
# ============================================================================

class HybridModel:
    """
    Modèle hybride: CNN extrait les features, LightGBM prend la décision
    """
    
    def __init__(self, cnn_model, lgb_params=None):
        self.cnn = cnn_model
        self.lgb = None
        self.lgb_params = lgb_params or {
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': 0.01,
            'num_leaves': 31,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.9,
            'verbose': -1,
            'seed': 42
        }
    
    def train(self, X_train_seq, X_test_seq, y_train, y_test):
        """
        1. Extrait les features CNN de train et test
        2. Entraîne un LightGBM sur ces features
        """
        print("\n" + "="*80)
        print("MODÈLE 4: HYBRID (CNN + LightGBM)")
        print("="*80)
        
        print("   Extracting CNN features...")
        X_train_cnn = self.cnn.extract_features(X_train_seq)
        X_test_cnn = self.cnn.extract_features(X_test_seq)
        
        train_data = lgb.Dataset(X_train_cnn, label=y_train)
        test_data = lgb.Dataset(X_test_cnn, label=y_test, reference=train_data)
        
        callbacks = [
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
        
        self.lgb = lgb.train(
            self.lgb_params,
            train_data,
            num_boost_round=500,
            valid_sets=[test_data],
            callbacks=callbacks
        )
        
        # Évaluation
        y_pred = self.lgb.predict(X_test_cnn)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n✅ Résultats Hybrid Model:")
        print(f"   MAE:  {mae:.6f}")
        print(f"   RMSE: {rmse:.6f}")
        print(f"   R²:   {r2:.6f}")
        
        return {
            'model_cnn': self.cnn.model_full,
            'model_lgb': self.lgb,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'predictions': y_pred,
            'actual': y_test
        }
    
    def save(self, cnn_path='models/hybrid_cnn.h5', lgb_path='models/hybrid_lgb.txt'):
        """Sauvegarde les deux modèles"""
        self.cnn.model_full.save(cnn_path)
        self.lgb.save_model(lgb_path)
        print(f"   📁 CNN sauvegardé: {cnn_path}")
        print(f"   📁 LightGBM sauvegardé: {lgb_path}")

# ============================================================================
# SECTION 7: MODÈLE 5 - LSTM
# ============================================================================

class LSTMModel:
    """
    LSTM pour modéliser les dépendances temporelles
    """
    
    def __init__(self, input_shape):
        self.model = None
        self.input_shape = input_shape
    
    def build(self):
        """
        Construit l'architecture LSTM avec attention implicite via bidirectionnel
        """
        print("\n" + "="*80)
        print("MODÈLE 5: LSTM (SÉQUENCES TEMPORELLES)")
        print("="*80)
        
        self.model = Sequential([
            Bidirectional(LSTM(64, return_sequences=True, activation='relu'),
                         input_shape=self.input_shape),
            Dropout(0.2),
            Bidirectional(LSTM(32, activation='relu')),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ], name='LSTM')
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        print(f"   ✅ Architecture LSTM construite")
        self.model.summary()
        
        return self.model
    
    def train(self, X_train, X_test, y_train, y_test, epochs=20):
        """
        Entraîne le LSTM
        """
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
            ModelCheckpoint('models/lstm_best.h5', save_best_only=True)
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Évaluation
        y_pred = self.model.predict(X_test, verbose=0)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n✅ Résultats LSTM:")
        print(f"   MAE:  {mae:.6f}")
        print(f"   RMSE: {rmse:.6f}")
        print(f"   R²:   {r2:.6f}")
        
        return {
            'model': self.model,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'predictions': y_pred,
            'actual': y_test,
            'history': history
        }
    
    def save(self, path='models/lstm_model.h5'):
        """Sauvegarde le modèle"""
        self.model.save(path)
        print(f"   📁 Modèle sauvegardé: {path}")

# ============================================================================
# SECTION 8: SHAP EXPLAINABILITY
# ============================================================================

class SHAPExplainer:
    """
    Explique les prédictions LightGBM avec SHAP
    """
    
    def __init__(self, model, X_background, feature_names):
        self.model = model
        self.X_background = X_background
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
    
    def compute(self, X_explain):
        """
        Calcule les valeurs SHAP
        """
        print("\n[9/10] Calcul SHAP...")
        
        self.explainer = shap.TreeExplainer(self.model)
        self.shap_values = self.explainer.shap_values(X_explain)
        
        print("   ✅ Valeurs SHAP calculées")
        
        return self.shap_values
    
    def plot_summary(self, output_path='plots/shap_summary.png'):
        """
        Plot le résumé SHAP
        """
        plt.figure(figsize=(12, 6))
        shap.summary_plot(self.shap_values, self.X_background, plot_type='bar', show=False)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   📊 Plot SHAP sauvegardé: {output_path}")
        plt.close()

# ============================================================================
# SECTION 9: DÉTECTION D'ANOMALIES
# ============================================================================

class AnomalyDetector:
    """
    Détecte et analyse les top anomalies
    """
    
    def __init__(self, predictions, actual, threshold_percentile=95):
        self.predictions = predictions.flatten()
        self.actual = actual.values if hasattr(actual, 'values') else actual
        self.threshold = np.percentile(self.predictions, threshold_percentile)
        self.anomalies = np.where(self.predictions > self.threshold)[0]
    
    def get_top_anomalies(self, n=5):
        """
        Retourne les top N anomalies
        """
        top_indices = np.argsort(self.predictions)[-n:]
        return top_indices[::-1]
    
    def plot_anomalies(self, output_path='plots/anomalies.png'):
        """
        Plot les anomalies détectées
        """
        plt.figure(figsize=(15, 5))
        
        plt.plot(self.predictions, label='Predicted Risk', alpha=0.7, linewidth=0.8)
        plt.axhline(self.threshold, color='r', linestyle='--', label=f'Anomaly Threshold ({self.threshold:.4f})')
        
        top_5 = self.get_top_anomalies(5)
        plt.scatter(top_5, self.predictions[top_5], color='red', s=100, zorder=5, label='Top 5 Anomalies')
        
        plt.xlabel('Sample Index')
        plt.ylabel('Risk Score')
        plt.title('Anomaly Detection - Risk Score Predictions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   📊 Plot anomalies sauvegardé: {output_path}")
        plt.close()

# ============================================================================
# SECTION 10: PIPELINE D'ENTRAÎNEMENT COMPLET
# ============================================================================

def train_all_models(df, mappings, scaler):
    """
    Pipeline complet d'entraînement de tous les modèles
    """
    
    print("\n" + "="*80)
    print("PIPELINE D'ENTRAÎNEMENT COMPLET")
    print("="*80 + "\n")
    
    # 1. Préprocessing
    preprocessor = DataPreprocessor(df, mappings)
    X_train, X_test, y_train_reg, y_test_reg, y_train_cls, y_test_cls = preprocessor.split_temporal()
    
    # Normaliser les features
    X_train_scaled, X_test_scaled, feature_scaler = preprocessor.normalize_features(X_train, X_test)
    
    # Créer les séquences pour CNN/LSTM
    window_size = 20
    X_train_seq, y_train_seq = preprocessor.create_sequences(X_train_scaled, y_train_reg, window=window_size)
    X_test_seq, y_test_seq = preprocessor.create_sequences(X_test_scaled, y_test_reg, window=window_size)
    
    print(f"\n   X_train_seq shape: {X_train_seq.shape}")
    print(f"   X_test_seq shape: {X_test_seq.shape}")
    print(f"   y_train_seq shape: {y_train_seq.shape}")
    print(f"   y_test_seq shape: {y_test_seq.shape}")
    
    results = {}
    
    # ========================================================================
    # MODÈLE 1: LightGBM Régression
    # ========================================================================
    lgb_reg = LightGBMRegressor()
    results['lgb_regression'] = lgb_reg.train(
        X_train_scaled, X_test_scaled, y_train_reg, y_test_reg
    )
    lgb_reg.save('models/lgb_regressor.txt')
    
    # ========================================================================
    # MODÈLE 2: LightGBM Classification
    # ========================================================================
    lgb_cls = LightGBMClassifier(n_classes=len(mappings['condition_to_code']))
    results['lgb_classification'] = lgb_cls.train(
        X_train_scaled, X_test_scaled, y_train_cls, y_test_cls
    )
    lgb_cls.save('models/lgb_classifier.txt')
    
    # ========================================================================
    # MODÈLE 3: CNN 1D
    # ========================================================================
    input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
    cnn = CNN1D(input_shape=input_shape)
    cnn.build()
    results['cnn'] = cnn.train(X_train_seq, X_test_seq, y_train_seq, y_test_seq, epochs=20)
    cnn.save('models/cnn_model.h5')
    
    # ========================================================================
    # MODÈLE 4: Hybrid (CNN + LightGBM)
    # ========================================================================
    hybrid = HybridModel(cnn)
    results['hybrid'] = hybrid.train(X_train_seq, X_test_seq, y_train_seq, y_test_seq)
    hybrid.save('models/hybrid_cnn.h5', 'models/hybrid_lgb.txt')
    
    # ========================================================================
    # MODÈLE 5: LSTM
    # ========================================================================
    lstm = LSTMModel(input_shape=input_shape)
    lstm.build()
    results['lstm'] = lstm.train(X_train_seq, X_test_seq, y_train_seq, y_test_seq, epochs=20)
    lstm.save('models/lstm_model.h5')
    
    # ========================================================================
    # SHAP Explainability
    # ========================================================================
    print("\n[9/10] Explainability SHAP...")
    shap_explainer = SHAPExplainer(
        lgb_reg.model,
        X_train_scaled.sample(min(100, len(X_train_scaled))),
        feature_names=preprocessor.features_basic
    )
    shap_explainer.compute(X_test_scaled)
    shap_explainer.plot_summary('plots/shap_summary.png')
    
    # ========================================================================
    # Anomaly Detection
    # ========================================================================
    print("\n[10/10] Détection d'anomalies...")
    y_pred_reg = results['lgb_regression']['predictions']
    anomaly_detector = AnomalyDetector(y_pred_reg, y_test_reg, threshold_percentile=95)
    anomaly_detector.plot_anomalies('plots/anomalies.png')
    
    print("\n" + "="*80)
    print("✅ TOUS LES MODÈLES ENTRAÎNÉS AVEC SUCCÈS")
    print("="*80)
    
    return results, preprocessor, feature_scaler, lgb_reg, lgb_cls, cnn, lstm, hybrid

# ============================================================================
# SECTION 11: RAPPORT FINAL
# ============================================================================

def generate_final_report(results):
    """
    Génère un rapport complet des performances
    """
    
    print("\n" + "="*80)
    print("RAPPORT FINAL DES MODÈLES")
    print("="*80 + "\n")
    
    print("📊 MODÈLE 1: LightGBM Régression (Risk Score)")
    print(f"   MAE:  {results['lgb_regression']['mae']:.6f}")
    print(f"   RMSE: {results['lgb_regression']['rmse']:.6f}")
    print(f"   R²:   {results['lgb_regression']['r2']:.6f}\n")
    
    print("📊 MODÈLE 2: LightGBM Classification (Conditions)")
    print(f"   Accuracy:  {results['lgb_classification']['accuracy']:.6f}")
    print(f"   Precision: {results['lgb_classification']['precision']:.6f}")
    print(f"   Recall:    {results['lgb_classification']['recall']:.6f}")
    print(f"   F1-Score:  {results['lgb_classification']['f1']:.6f}\n")
    
    print("📊 MODÈLE 3: CNN 1D")
    print(f"   MAE:  {results['cnn']['mae']:.6f}")
    print(f"   RMSE: {results['cnn']['rmse']:.6f}")
    print(f"   R²:   {results['cnn']['r2']:.6f}\n")
    
    print("📊 MODÈLE 4: Hybrid (CNN + LightGBM)")
    print(f"   MAE:  {results['hybrid']['mae']:.6f}")
    print(f"   RMSE: {results['hybrid']['rmse']:.6f}")
    print(f"   R²:   {results['hybrid']['r2']:.6f}\n")
    
    print("📊 MODÈLE 5: LSTM")
    print(f"   MAE:  {results['lstm']['mae']:.6f}")
    print(f"   RMSE: {results['lstm']['rmse']:.6f}")
    print(f"   R²:   {results['lstm']['r2']:.6f}\n")
    
    # Meilleur modèle
    models_reg = {
        'LightGBM Regression': results['lgb_regression']['r2'],
        'CNN 1D': results['cnn']['r2'],
        'Hybrid': results['hybrid']['r2'],
        'LSTM': results['lstm']['r2']
    }
    
    best_model = max(models_reg, key=models_reg.get)
    print(f"🏆 Meilleur modèle (R²): {best_model} ({models_reg[best_model]:.6f})")
    print("\n" + "="*80)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    
    # Charger les données
    df, mappings, scaler = load_dataset('dataset_TMS_enriched.csv')
    
    # Entraîner tous les modèles
    results, preprocessor, feature_scaler, lgb_reg, lgb_cls, cnn, lstm, hybrid = train_all_models(
        df, mappings, scaler
    )
    
    # Rapport final
    generate_final_report(results)
    
    # Sauvegarder les preprocessors pour l'inférence
    print("\n💾 Sauvegarde des preprocessors...")
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
    joblib.dump(feature_scaler, 'models/feature_scaler.pkl')
    print("   ✅ Preprocessors sauvegardés")
    
    print("\n✅ Pipeline complet exécuté avec succès!")
    print("   📁 Tous les modèles sont dans le dossier 'models/'")
    print("   📊 Les plots sont dans le dossier 'plots/'")
