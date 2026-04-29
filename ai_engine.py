# ai_engine.py
import json
import joblib
import lightgbm as lgb
import numpy as np
import shap
from pathlib import Path

class AIModels:
    def __init__(self, model_dir='models'):
        self.model_dir = Path(model_dir)
        self.ready = False
        try:
            self._load_models()
            self.ready = True
        except Exception as e:
            print(f"[ERROR] Failed to load AI models: {e}")

    def _load_models(self):
        meta_path = self.model_dir / 'model_metadata.json'
        self.meta = json.load(open(meta_path))

        # Expose convenience attributes used by ReportGenerator
        self.feature_cols = self.meta.get('feature_cols', [])
        self.seq_len      = int(self.meta.get('seq_len', 60))

        # Main 10-day risk prediction model
        self.risk_model = lgb.Booster(model_file=str(self.model_dir / 'lgbm_risk_10d.txt'))
        self.explainer = shap.TreeExplainer(self.risk_model)

        # Isolation Forest for anomaly score
        self.iso_forest = joblib.load(self.model_dir / 'isolation_forest.pkl')
        self.scaler_if = joblib.load(self.model_dir / 'scaler_if.pkl')

        # Load all 5 anomaly models (some may be dummy models)
        self.anomaly_models = {}
        for col in self.meta.get('anomaly_cols', []):
            model_path = self.model_dir / f'lgbm_{col}.txt'
            if model_path.exists():
                self.anomaly_models[col] = lgb.Booster(model_file=str(model_path))
                print(f"[OK] Loaded anomaly model: {col}")
            else:
                print(f"[WARN] Missing anomaly model: {col}")

        print(f"[OK] Ergo Sensor AI Models loaded | Risk model: OK | Anomaly models: {len(self.anomaly_models)}/5")

    def predict(self, features_dict):
        """Main prediction function - expects a dict with all feature columns"""
        try:
            # Prepare features for risk model
            X = np.array([[features_dict.get(c, 0.0) for c in self.meta['feature_cols']]],
                         dtype=np.float32)

            # Risk prediction (10-day forecast)
            risk_10d = float(self.risk_model.predict(X)[0])

            # SHAP critical feature extraction
            shap_values = self.explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_vals = shap_values[1][0]  # If binary classification
            else:
                shap_vals = shap_values[0]
            
            top_idx = np.argmax(np.abs(shap_vals))
            critical_joint = self.meta['feature_cols'][top_idx]

            # Anomaly score using Isolation Forest
            X_if = np.array([[features_dict.get(c, 0.0) for c in self.meta.get('if_features', [])]],
                            dtype=np.float32)
            if X_if.shape[1] > 0:
                raw_score = self.iso_forest.decision_function(self.scaler_if.transform(X_if))[0]
                anomaly_score = round(float(1 - raw_score), 3)
            else:
                anomaly_score = 0.0

            # Top-5 anomaly probabilities
            anomaly_probs = {}
            for col, name in zip(self.meta.get('anomaly_cols', []), self.meta.get('anomaly_names', [])):
                if col in self.anomaly_models:
                    prob = float(self.anomaly_models[col].predict(X)[0])
                    anomaly_probs[name] = round(prob, 3)
                else:
                    anomaly_probs[name] = 0.0

            # Determine risk level
            if risk_10d >= 0.8:
                level = 'HIGH'
            elif risk_10d >= 0.6:
                level = 'MODERATE'
            elif risk_10d >= 0.4:
                level = 'LOW'
            else:
                level = 'SAFE'

            return {
                'risk_10d': round(risk_10d, 3),
                'risk_level': level,
                'anomaly_score': anomaly_score,
                'anomaly_probs': anomaly_probs,
                'top_anomaly': max(anomaly_probs, key=anomaly_probs.get, default=None),
                'critical_joint': critical_joint,
                'current_risk': round(features_dict.get('global_risk_score', 0.0), 3)
            }

        except Exception as e:
            print(f"[ERROR] AI Prediction Error: {e}")
            return {
                'risk_10d': 0.0,
                'risk_level': 'ERROR',
                'anomaly_score': 0.0,
                'anomaly_probs': {},
                'top_anomaly': None,
                'critical_joint': None,
                'error': str(e)
            }