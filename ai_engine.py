# ai_engine.py  — version 2.0 (corps entier, 38 features)
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

        self.feature_cols = self.meta.get('feature_cols', [])
        self.seq_len      = int(self.meta.get('seq_len', 60))

        # ── Regression risk_score ──────────────────────────────────────────
        self.risk_model = lgb.Booster(
            model_file=str(self.model_dir / 'lgb_regressor.txt')
        )
        self.explainer = shap.TreeExplainer(self.risk_model)

        # ── Classification condition ───────────────────────────────────────
        self.condition_model = lgb.Booster(
            model_file=str(self.model_dir / 'lgb_classifier.txt')
        )
        self.condition_map = {
            v: k for k, v in self.meta.get('condition_to_code', {}).items()
        }

        # ── Classification severite ────────────────────────────────────────
        self.severity_model = lgb.Booster(
            model_file=str(self.model_dir / 'lgb_severity.txt')
        )
        self.severity_map = {v: k for k, v in self.meta.get('severity_to_code', {}).items()}

        # ── Isolation Forest anomalie globale ─────────────────────────────
        self.iso_forest = joblib.load(self.model_dir / 'isolation_forest.pkl')
        self.scaler_if  = joblib.load(self.model_dir / 'scaler_if.pkl')

        # ── Anomaly models (5 articulations) ─────────────────────────────
        self.anomaly_models = {}
        for col in self.meta.get('anomaly_cols', []):
            model_path = self.model_dir / f'lgbm_{col}.txt'
            if model_path.exists():
                self.anomaly_models[col] = lgb.Booster(model_file=str(model_path))
                print(f"[OK] Loaded anomaly model: {col}")
            else:
                print(f"[WARN] Missing anomaly model: {col}")

        n_anom = len(self.anomaly_models)
        print(f"[OK] Ergo Sensor AI Models loaded | Risk model: OK | Anomaly models: {n_anom}/5")

    # ─────────────────────────────────────────────────────────────────────────
    def predict(self, features_dict):
        """
        Prediction complete a partir d'un dict de features.
        Retourne risk_score, condition, severite, anomalie, SHAP top joint.
        """
        try:
            X = np.array(
                [[features_dict.get(c, 0.0) for c in self.feature_cols]],
                dtype=np.float32
            )

            # ── Risk score (regression 0-1) ──────────────────────────────
            risk_score = float(self.risk_model.predict(X)[0])
            risk_score = float(np.clip(risk_score, 0.0, 1.0))

            # ── SHAP critical joint ───────────────────────────────────────
            shap_values = self.explainer.shap_values(X)
            shap_vals   = shap_values[0] if not isinstance(shap_values, list) else shap_values[0][0]
            top_idx     = int(np.argmax(np.abs(shap_vals)))
            critical_joint = (
                self.feature_cols[top_idx] if top_idx < len(self.feature_cols) else None
            )

            # ── Condition ─────────────────────────────────────────────────
            cond_probs = self.condition_model.predict(X)[0]  # shape (n_classes,)
            cond_code  = int(np.argmax(cond_probs))
            condition  = self.condition_map.get(cond_code, 'unknown')

            # ── Severite ─────────────────────────────────────────────────
            sev_probs = self.severity_model.predict(X)[0]
            sev_code  = int(np.argmax(sev_probs))
            severity  = self.severity_map.get(sev_code, 'low')

            import pandas as pd
            if_cols = self.meta.get('if_features', self.feature_cols)
            X_if = pd.DataFrame(
                [[features_dict.get(c, 0.0) for c in if_cols]],
                columns=if_cols
            )
            raw_if_score  = self.iso_forest.decision_function(self.scaler_if.transform(X_if))[0]
            anomaly_score = round(float(1.0 - raw_if_score), 3)

            # ── Anomaly probs per articulation ────────────────────────────
            anomaly_probs = {}
            anomaly_cols  = self.meta.get('anomaly_cols', [])
            anomaly_names = self.meta.get('anomaly_names', anomaly_cols)
            for col, name in zip(anomaly_cols, anomaly_names):
                if col in self.anomaly_models:
                    prob = self.anomaly_models[col].predict(X)[0]
                    anomaly_probs[name] = round(float(prob), 3)
                else:
                    anomaly_probs[name] = 0.0

            # ── Risk level ────────────────────────────────────────────────
            if risk_score >= 0.80:
                level = 'HIGH'
            elif risk_score >= 0.60:
                level = 'MODERATE'
            elif risk_score >= 0.40:
                level = 'LOW'
            else:
                level = 'SAFE'

            return {
                'risk_10d':      round(risk_score, 3),
                'risk_level':    level,
                'condition':     condition,
                'severity':      severity,
                'anomaly_score': anomaly_score,
                'anomaly_probs': anomaly_probs,
                'top_anomaly':   max(anomaly_probs, key=anomaly_probs.get, default=None),
                'critical_joint': critical_joint,
                'current_risk':  round(features_dict.get('global_risk_score', risk_score), 3),
                'body_parts':    self.meta.get('body_parts', []),
            }

        except Exception as e:
            print(f"[ERROR] AI Prediction Error: {e}")
            return {
                'risk_10d':      0.0,
                'risk_level':    'ERROR',
                'condition':     'unknown',
                'severity':      'low',
                'anomaly_score': 0.0,
                'anomaly_probs': {},
                'top_anomaly':   None,
                'critical_joint': None,
                'error':         str(e),
            }