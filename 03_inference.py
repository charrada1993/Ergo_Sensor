"""
=============================================================================
ÉTAPE 3 : INFÉRENCE - UTILISATION DES MODÈLES POUR PRÉDICTIONS
=============================================================================
Auteur: Ingénieur IA Senior
Date: 2025
Version: 1.0 (Production-Ready)

✅ Charge les modèles entraînés et fait des prédictions
✅ Explique les prédictions avec SHAP
✅ Gère plusieurs formats d'entrée
✅ Prêt pour déploiement API
=============================================================================
"""

import pandas as pd
import numpy as np
import json
import pickle
import joblib
import warnings
from datetime import datetime

# ML & DL
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import load_model
import shap

warnings.filterwarnings('ignore')

# ============================================================================
# SECTION 1: LOADER DE MODÈLES
# ============================================================================

class ModelLoader:
    """
    Charge tous les modèles entraînés et les métadonnées
    """
    
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.models = {}
        self.preprocessor = None
        self.feature_scaler = None
        self.mappings = None
        self.code_to_condition = None
        self.code_to_severity = None
        self.code_to_location = None
    
    def load_all(self):
        """
        Charge tous les modèles et métadonnées
        """
        print("[1/3] Chargement de tous les modèles...")
        
        # Charger les modèles LightGBM
        self.models['lgb_regressor'] = lgb.Booster(model_file=f'{self.models_dir}/lgb_regressor.txt')
        self.models['lgb_classifier'] = lgb.Booster(model_file=f'{self.models_dir}/lgb_classifier.txt')
        print("   ✅ Modèles LightGBM chargés")
        
        # Charger les modèles Deep Learning
        self.models['cnn'] = load_model(f'{self.models_dir}/cnn_model.h5')
        self.models['lstm'] = load_model(f'{self.models_dir}/lstm_model.h5')
        self.models['hybrid_cnn'] = load_model(f'{self.models_dir}/hybrid_cnn.h5')
        self.models['hybrid_lgb'] = lgb.Booster(model_file=f'{self.models_dir}/hybrid_lgb.txt')
        print("   ✅ Modèles Deep Learning chargés")
        
        # Charger les preprocessors
        self.preprocessor = joblib.load(f'{self.models_dir}/preprocessor.pkl')
        self.feature_scaler = joblib.load(f'{self.models_dir}/feature_scaler.pkl')
        print("   ✅ Preprocessors chargés")
        
        # Charger les métadonnées
        with open('condition_mappings.json', 'r') as f:
            self.mappings = json.load(f)
        
        # Créer les reverse mappings
        condition_to_code = self.mappings['condition_to_code']
        self.code_to_condition = {v: k for k, v in condition_to_code.items()}
        
        severity_to_code = self.mappings['severity_to_code']
        self.code_to_severity = {v: k for k, v in severity_to_code.items()}
        
        location_to_code = self.mappings['location_to_code']
        self.code_to_location = {v: k for k, v in location_to_code.items()}
        
        print("   ✅ Mappings chargés\n")
        
        return self.models

# ============================================================================
# SECTION 2: PIPELINES DE PRÉDICTION
# ============================================================================

class PredictionPipeline:
    """
    Pipeline de prédiction unifiée pour tous les modèles
    """
    
    def __init__(self, model_loader):
        self.loader = model_loader
        self.models = model_loader.models
        self.preprocessor = model_loader.preprocessor
        self.feature_scaler = model_loader.feature_scaler
        self.features = self.preprocessor.features_basic
    
    def preprocess_input(self, input_df):
        """
        Préprocesse les données d'entrée
        """
        # Copier le dataframe
        df = input_df.copy()
        
        # Vérifier que toutes les features sont présentes
        missing_features = set(self.features) - set(df.columns)
        if missing_features:
            raise ValueError(f"Features manquantes: {missing_features}")
        
        # Extraire les features
        X = df[self.features].copy()
        
        # Normaliser avec le scaler d'entraînement
        X_scaled = pd.DataFrame(
            self.feature_scaler.transform(X),
            columns=self.features,
            index=X.index
        )
        
        return X_scaled
    
    def predict_regression(self, input_df):
        """
        Prédit le risk_score avec le meilleur modèle de régression
        """
        print("[2/3] Prédiction du risk_score...")
        
        X = self.preprocess_input(input_df)
        
        # Utiliser LightGBM (généralement plus stable)
        risk_scores = self.models['lgb_regressor'].predict(X)
        
        return risk_scores
    
    def predict_classification(self, input_df):
        """
        Classifie les conditions
        """
        print("[2/3] Classification des conditions...")
        
        X = self.preprocess_input(input_df)
        
        # Prédiction avec probabilités
        condition_codes = self.models['lgb_classifier'].predict(X)
        condition_probs = self.models['lgb_classifier'].predict_proba(X)
        
        # Convertir les codes en noms
        conditions = np.array([self.loader.code_to_condition[int(c)] for c in condition_codes])
        
        return conditions, condition_codes, condition_probs
    
    def predict_severity(self, risk_scores):
        """
        Déduit la sévérité du risk_score
        """
        severity = []
        for score in risk_scores:
            if score > 0.85:
                severity.append('high')
            elif score > 0.65:
                severity.append('medium')
            else:
                severity.append('low')
        
        return np.array(severity)
    
    def predict_location(self, input_df):
        """
        Détecte la localisation principale du problème
        """
        stress_by_location = {
            'neck': input_df['neck'].values,
            'trunk': input_df['trunk'].values,
            'shoulder': input_df['shoulder'].values,
            'wrist': input_df['wrist'].values
        }
        
        locations = []
        for i in range(len(input_df)):
            max_location = max(
                {k: v[i] for k, v in stress_by_location.items()},
                key=lambda k: stress_by_location[k][i]
            )
            locations.append(max_location)
        
        return np.array(locations)
    
    def predict_all(self, input_df):
        """
        Prédiction complète:
        - Risk score
        - Condition principale
        - Sévérité
        - Localisation
        - Probabilités par condition
        """
        print("[2/3] Prédiction complète...")
        
        # Risk score
        risk_scores = self.predict_regression(input_df)
        
        # Conditions
        conditions, condition_codes, condition_probs = self.predict_classification(input_df)
        
        # Sévérité
        severity = self.predict_severity(risk_scores)
        
        # Localisation
        locations = self.predict_location(input_df)
        
        return {
            'risk_score': risk_scores,
            'condition': conditions,
            'condition_code': condition_codes,
            'condition_probabilities': condition_probs,
            'severity': severity,
            'location': locations,
            'timestamp': datetime.now().isoformat()
        }

# ============================================================================
# SECTION 3: EXPLAINABILITY AVEC SHAP
# ============================================================================

class ExplainabilityEngine:
    """
    Explique les prédictions avec SHAP
    """
    
    def __init__(self, model_loader, pipeline):
        self.loader = model_loader
        self.pipeline = pipeline
        self.explainer = None
    
    def initialize(self, background_data=None, n_background=100):
        """
        Initialise l'explainer SHAP
        """
        print("[3/3] Initialisation SHAP...")
        
        if background_data is None:
            # Charger un échantillon de données d'entraînement
            df_train = pd.read_csv('dataset_TMS_enriched.csv')
            background_data = df_train[self.pipeline.features].sample(n=min(n_background, len(df_train)))
        
        # Normaliser les données de background
        background_scaled = pd.DataFrame(
            self.pipeline.feature_scaler.transform(background_data),
            columns=self.pipeline.features
        )
        
        # Créer l'explainer
        self.explainer = shap.TreeExplainer(self.loader.models['lgb_regressor'])
        
        print("   ✅ SHAP initialiser\n")
        
        return self.explainer
    
    def explain_prediction(self, X_input):
        """
        Explique une prédiction spécifique
        """
        if self.explainer is None:
            self.initialize()
        
        X_scaled = pd.DataFrame(
            self.pipeline.feature_scaler.transform(X_input),
            columns=self.pipeline.features
        )
        
        shap_values = self.explainer.shap_values(X_scaled)
        expected_value = self.explainer.expected_value
        
        return shap_values, expected_value, X_scaled
    
    def get_feature_importance(self, shap_values):
        """
        Calcule l'importance des features à partir des valeurs SHAP
        """
        # Moyenne absolue des valeurs SHAP
        importance = np.abs(shap_values).mean(axis=0)
        
        # Trier
        feature_importance = pd.DataFrame({
            'feature': self.pipeline.features,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance

# ============================================================================
# SECTION 4: FORMATAGE DES RÉSULTATS
# ============================================================================

class ResultFormatter:
    """
    Formate les résultats pour différents cas d'usage
    """
    
    @staticmethod
    def to_json(predictions, index=0):
        """
        Formate une prédiction en JSON
        """
        return {
            'risk_score': float(predictions['risk_score'][index]),
            'condition': str(predictions['condition'][index]),
            'severity': str(predictions['severity'][index]),
            'location': str(predictions['location'][index]),
            'condition_probabilities': {
                'condition': str(condition),
                'probability': float(prob)
            },
            'timestamp': predictions['timestamp']
        }
    
    @staticmethod
    def to_dataframe(predictions):
        """
        Retourne les résultats sous forme de DataFrame
        """
        return pd.DataFrame({
            'risk_score': predictions['risk_score'],
            'condition': predictions['condition'],
            'severity': predictions['severity'],
            'location': predictions['location']
        })
    
    @staticmethod
    def to_report(predictions, explainer=None, shap_values=None, index=0):
        """
        Génère un rapport texte détaillé
        """
        report = f"""
╔════════════════════════════════════════════════════════════════════════╗
║                    RAPPORT DE PRÉDICTION TMS                           ║
╚════════════════════════════════════════════════════════════════════════╝

📊 ANALYSE PRINCIPALE
{'─' * 76}
Risk Score:        {predictions['risk_score'][index]:.4f} (0-1)
Condition:         {predictions['condition'][index]}
Sévérité:          {predictions['severity'][index].upper()}
Localisation:      {predictions['location'][index].upper()}
Timestamp:         {predictions['timestamp']}

📈 INTERPRÉTATION SÉVÉRITÉ
{'─' * 76}
"""
        score = predictions['risk_score'][index]
        if score > 0.85:
            report += "🔴 CRITIQUE: Risque très élevé d'atteinte musculo-squelettique\n"
            report += "   → Intervention médicale recommandée\n"
            report += "   → Modifier immédiatement la posture/ergonomie\n"
        elif score > 0.65:
            report += "🟠 ÉLEVÉ: Risque modéré à élevé\n"
            report += "   → Suivi médical recommandé\n"
            report += "   → Prendre des pauses régulières\n"
        else:
            report += "🟢 BAS: Risque acceptable\n"
            report += "   → Continuer le suivi ergonomique\n"
        
        report += f"""
📍 LOCALISATION
{'─' * 76}
Zone affectée: {predictions['location'][index]}

⚠️  CONDITIONS DÉTECTÉES
{'─' * 76}
"""
        # Ajouter les conditions avec probabilités
        probs = predictions['condition_probabilities'][index]
        for i, prob in enumerate(sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:5]):
            idx, p = prob
            if p > 0.1:  # Seuil minimum
                report += f"   • Condition {idx}: {p*100:.1f}%\n"
        
        report += f"""
{'─' * 76}
Rapport généré le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
╚════════════════════════════════════════════════════════════════════════╝
"""
        return report

# ============================================================================
# SECTION 5: EXEMPLES D'UTILISATION
# ============================================================================

def example_single_prediction():
    """
    Exemple 1: Prédiction simple sur un seul échantillon
    """
    print("\n" + "="*80)
    print("EXEMPLE 1: Prédiction Simple")
    print("="*80 + "\n")
    
    # Charger les modèles
    loader = ModelLoader()
    loader.load_all()
    
    # Pipeline de prédiction
    pipeline = PredictionPipeline(loader)
    
    # Créer un exemple d'entrée
    example_input = pd.DataFrame({
        'neck': [35.0],
        'trunk': [50.0],
        'shoulder': [60.0],
        'wrist': [25.0],
        'neck_vel': [2.5],
        'trunk_vel': [3.0],
        'shoulder_vel': [2.0],
        'wrist_vel': [1.5],
        'neck_duration': [10],
        'trunk_duration': [15],
        'shoulder_duration': [8],
        'wrist_duration': [5],
        'neck_freq': [3],
        'trunk_freq': [4]
    })
    
    # Prédire
    predictions = pipeline.predict_all(example_input)
    
    # Formatter les résultats
    formatter = ResultFormatter()
    report = formatter.to_report(predictions)
    print(report)
    
    return predictions

def example_batch_prediction():
    """
    Exemple 2: Prédiction sur un batch de données
    """
    print("\n" + "="*80)
    print("EXEMPLE 2: Prédiction en Batch")
    print("="*80 + "\n")
    
    # Charger les modèles
    loader = ModelLoader()
    loader.load_all()
    
    # Pipeline de prédiction
    pipeline = PredictionPipeline(loader)
    
    # Charger un batch du dataset
    df = pd.read_csv('dataset_TMS_enriched.csv')
    batch = df[pipeline.features].iloc[:10].copy()  # Premiers 10 samples
    
    # Prédire
    predictions = pipeline.predict_all(batch)
    
    # Formatter en DataFrame
    formatter = ResultFormatter()
    results_df = formatter.to_dataframe(predictions)
    
    print("✅ Résultats du batch:\n")
    print(results_df.head(10).to_string())
    
    return predictions

def example_with_explainability():
    """
    Exemple 3: Prédiction avec explainabilité SHAP
    """
    print("\n" + "="*80)
    print("EXEMPLE 3: Prédiction avec SHAP")
    print("="*80 + "\n")
    
    # Charger les modèles
    loader = ModelLoader()
    loader.load_all()
    
    # Pipelines
    pipeline = PredictionPipeline(loader)
    explainer_engine = ExplainabilityEngine(loader, pipeline)
    explainer_engine.initialize()
    
    # Créer un exemple d'entrée
    example_input = pd.DataFrame({
        'neck': [40.0],
        'trunk': [55.0],
        'shoulder': [70.0],
        'wrist': [28.0],
        'neck_vel': [3.0],
        'trunk_vel': [3.5],
        'shoulder_vel': [2.5],
        'wrist_vel': [2.0],
        'neck_duration': [12],
        'trunk_duration': [18],
        'shoulder_duration': [10],
        'wrist_duration': [6],
        'neck_freq': [4],
        'trunk_freq': [5]
    })
    
    # Prédiction complète
    predictions = pipeline.predict_all(example_input)
    
    # Explainability
    shap_values, expected_value, X_scaled = explainer_engine.explain_prediction(example_input)
    feature_importance = explainer_engine.get_feature_importance(shap_values)
    
    print(f"✅ Prédiction: Risk Score = {predictions['risk_score'][0]:.4f}")
    print(f"   Condition: {predictions['condition'][0]}")
    print(f"   Sévérité: {predictions['severity'][0]}\n")
    
    print("📊 Feature Importance (Top 5):\n")
    print(feature_importance.head(5).to_string())
    
    return predictions, feature_importance

# ============================================================================
# SECTION 6: CLASSE API (pour déploiement)
# ============================================================================

class InferenceAPI:
    """
    API de prédiction pour déploiement en production
    """
    
    def __init__(self):
        self.loader = ModelLoader()
        self.loader.load_all()
        self.pipeline = PredictionPipeline(self.loader)
        self.explainer = ExplainabilityEngine(self.loader, self.pipeline)
        self.explainer.initialize()
        self.formatter = ResultFormatter()
        print("✅ API initialisée et prête\n")
    
    def predict(self, data_dict, explain=False):
        """
        Endpoint principal de prédiction
        
        Input:
            data_dict: dict ou pandas Series avec les features
            explain: bool, si True, retourner aussi les explications SHAP
        
        Output:
            dict avec les prédictions
        """
        # Convertir en DataFrame si nécessaire
        if isinstance(data_dict, dict):
            data_df = pd.DataFrame([data_dict])
        else:
            data_df = data_dict
        
        # Prédire
        predictions = self.pipeline.predict_all(data_df)
        
        # Ajouter les explications si demandé
        if explain:
            shap_vals, exp_val, X_scaled = self.explainer.explain_prediction(data_df)
            feat_imp = self.explainer.get_feature_importance(shap_vals)
            predictions['feature_importance'] = feat_imp.to_dict('records')
        
        return predictions
    
    def predict_batch(self, data_df):
        """
        Batch prediction
        """
        return self.pipeline.predict_all(data_df)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*80)
    print("EXEMPLES D'INFÉRENCE - MODÈLES ENTRAÎNÉS")
    print("="*80 + "\n")
    
    # Exemple 1: Prédiction simple
    predictions_1 = example_single_prediction()
    
    # Exemple 2: Batch prediction
    predictions_2 = example_batch_prediction()
    
    # Exemple 3: Avec explainabilité
    predictions_3, feature_imp = example_with_explainability()
    
    print("\n" + "="*80)
    print("✅ TOUS LES EXEMPLES EXÉCUTÉS AVEC SUCCÈS")
    print("="*80)
    
    # Montrer comment utiliser l'API
    print("\n💡 UTILISATION DE L'API EN PRODUCTION:\n")
    print("""
# Initialiser l'API
api = InferenceAPI()

# Prédiction simple
result = api.predict({
    'neck': 35.0,
    'trunk': 50.0,
    'shoulder': 60.0,
    'wrist': 25.0,
    # ... autres features
}, explain=True)

# Batch prediction
results = api.predict_batch(data_df)
    """)
