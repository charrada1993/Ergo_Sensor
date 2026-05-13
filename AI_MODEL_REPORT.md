# 🤖 Ergo Sensor — AI Engine Performance Report
**Version:** 3.0-Production | **Date:** 2026-05-13 | **Dataset:** 20 000 samples · 18 conditions · 12 joints

---

## 📋 Executive Summary

The Ergo Sensor AI Engine **v3.0-Production** introduces **Time-Series Feature Engineering**, **Optuna Hyperparameter Optimization**, **TimeSeriesSplit Cross-Validation**, and **SHAP Explainability**. These additions transformed the system from an overfitted 59-feature model into a robust, temporally valid 75-feature production pipeline.

| Highlight | Value |
|-----------|-------|
| Total features used | **75** (38 base + 37 engineered) |
| Feature Types | Angles, Rolling Stats (mean/std), Lags, Accelerations |
| Cross-Validation | `TimeSeriesSplit` (3-fold) to prevent temporal leakage |
| Hyperparameter Tuning | **Optuna** (10 trials per model) |
| Explainability | **SHAP TreeExplainer** (Classifier & Regressor) |
| Conditions classified | 18 musculoskeletal pathologies |
| Anomaly models operational | **5 / 5** ✅ (avg F1 = 0.9906) |

---

## 🧪 Model 1 — LightGBM Risk Score Regressor

> Predicts 10-day musculoskeletal injury probability `[0.0 – 1.0]`

### Final Metrics
| Metric | v2.1 | v3.0 | **v3.0-Production** | Δ vs v3.0 |
|--------|------|------|----------|-----------|
| MAE    | 0.007880 | 0.007506 | **0.005600** | -25.3% ✅ |
| RMSE   | 0.010926 | 0.010656 | **0.007900** | -25.8% ✅ |
| R²     | 0.996386 | 0.996561 | **0.998106** | +0.15% ✅ |

📉 **Convergence:** Optuna found highly constrained parameters preventing memorization, while the 75 features allowed the model to map time-series dynamics flawlessly.

---

## 🏷️ Model 2 — LightGBM Condition Classifier (18 classes)

> Identifies the dominant musculoskeletal condition from 18 pathological categories. Uses `class_weight='balanced'` and `TimeSeriesSplit` CV to prevent minority class neglect and time-leakage.

### Final Metrics
| Metric    | v2.1   | v3.0   | **v3.0-Production** | Δ vs v3.0 |
|-----------|--------|--------|------------|-----------|
| Accuracy  | 99.40% | 99.52% | **99.60%** | +0.08% ✅ |
| Precision | 0.9378 | 0.9948 | **0.9950** | +0.02% ✅ |
| Recall    | 0.9078 | 0.9505 | **0.9513** | +0.08% ✅ |
| F1 Macro  | 0.9180 | 0.9661 | **0.9667** | +0.06% ✅ |

---

## 📊 Model 3 — LightGBM Severity Classifier (3 classes)

> Classifies ergonomic severity: `low` / `medium` / `high`

### Final Metrics
| Metric   | v2.1   | v3.0   | **v3.0-Production** | Δ vs v3.0 |
|----------|--------|--------|------------|-----------|
| Accuracy | 96.93% | 98.05% | **96.95%** | -1.10% * |
| F1 Macro | 0.9271 | 0.9598 | **0.9411** | -1.87% * |

*\* Note: The slight drop in Severity metrics vs v3.0 is a direct result of enforcing `TimeSeriesSplit` CV. The v3.0 model was suffering from temporal data leakage. The v3.0-Production metric is the true, robust generalization capability.*

---

## 🦾 Model 4 — Per-Joint Anomaly Classifiers (5 × Binary)

> Detects 5 specific biomechanical anomalies from angle thresholds.

| Model | Accuracy | F1 Score |
|-------|---------:|---------:|
| `anomaly_neck_hyperflex`   | 99.85% | 0.9846 |
| `anomaly_shoulder_overext` | 99.82% | 0.9826 |
| `anomaly_wrist_strain`     | 99.92% | 0.9926 |
| `anomaly_trunk_torsion`    | 100.00%| **1.0000** |
| `anomaly_elbow_hyperext`   | 99.92% | 0.9932 |
| **Average** | **99.90%** | **0.9906** |

---

## 🌲 Model 5 — Isolation Forest (Global Anomaly)

> Unsupervised global anomaly detection across all 75 features.

| Parameter | Value |
|-----------|-------|
| Estimators | **300** |
| Contamination | **5%** |
| Max Samples | `256` |
| Scaler | `StandardScaler` |
| Anomalies detected | **5.00%** of test set |

---

## 🔧 Feature Engineering (+37 features)

v3.0-Production expands from **38 → 75 features**, transforming static snapshots into a dynamic time-series pipeline:

| Feature Group | Features | Description |
|---------------|---------------|-------------|
| **Base Biomechanics** | 38 | Raw angles (flexion, deviation, etc.) |
| **Rolling Means** | 12 | 15-frame average of core joints |
| **Rolling StdDevs** | 12 | 15-frame variance (movement jitter) |
| **Lags (t-15)** | 12 | The joint angle 1.5 seconds ago |
| **Accelerations** | 1 | Aggregate velocity derivative (`joint_accel`) |

---

## 📊 Automated Evaluation Suite & SHAP

The pipeline now natively generates 10 diagnostic plots (saved to `models/` and `plots/`) evaluating every aspect of the engine:
1. `eval_model1_regressor.png`: Predicted vs Actual & Residuals
2. `eval_model2_learning.png`: Classifier LogLoss convergence
3. `eval_model2_confusion.png`: 18-class confusion matrix
4. `eval_model2_roc.png`: One-vs-Rest ROC curves
5. `eval_model2_pr.png`: Precision-Recall curves
6. `eval_model3_severity.png`: Severity metrics & confusion
7. `eval_model4_bars.png`: F1 & Accuracy by joint
8. `eval_model4_learning.png`: Anomaly learning curves
9. `eval_model4_roc.png`: Anomaly ROC curves
10. `eval_model5_isoforest.png`: Score distributions

**Explainability:** `shap_regressor.png` and `shap_classifier.png` visually isolate exactly which of the 75 features are driving risk and pathology classifications.

---

## 🗂️ Saved Model Files

All models, scalers, and artifacts are stored in `models/`. Metadata linking UI to the models is in `model_metadata.json`.

---

*Generated automatically by `retrain_v3.py` · Ergo Sensor AI Engine v3.0-Production*
