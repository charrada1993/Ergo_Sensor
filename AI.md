# 🧠 Ergo Sensor — AI Intelligence & Biomechanical Modeling

This document provides a deep dive into the Artificial Intelligence core of the **Ergo Sensor** system. It explains the scientific rationale, the model architectures, and the data pipeline used to prevent Musculoskeletal Disorders (MSD).

---

## 🔬 Why This Model Choice?

We selected an **Ensemble Learning** approach based on **LightGBM** and **Isolation Forest** for several technical reasons:

1.  **Temporal Sequences**: Ergo Sensor v3.0-Production relies on time-series feature engineering (rolling windows, lags) and `TimeSeriesSplit` cross-validation to capture dynamic postural history rather than static snapshots.
2.  **Optuna HPO**: Hyperparameter Optimization (Optuna) ensures that models generalize to unseen workers rather than memorizing training data.
3.  **Handling Imbalance**: Postural anomalies are rare. LightGBM's `class_weight='balanced'` allows us to detect rare injuries effectively.
4.  **Unsupervised Hybrid**: By combining Isolation Forest (Unsupervised) with LightGBM (Supervised), we can detect both *known* disorders (like Neck Hyperflexion) and *unknown* dangerous movements.

---

## 📊 The Dataset: `dataset_TMS_enriched.csv`

The model was trained on a high-quality dataset specifically engineered for Musculoskeletal Disorder (MSD) research:

- **Volume**: ~20,000 data points collected at 10Hz.
- **Input Features (75)**: 
    - 12 raw joint angles (Clinical standard).
    - 24 temporal statistics (Rolling Means and StdDevs).
    - 12 lag features (Postural history).
    - 14 dynamic & proxy features (Velocity, Energy).
    - 5 bilateral asymmetry deltas.
    - 2 composite load scores.
    - 6 raw angle overlays, posture flags, and accelerations.
- **Target Classes**: 
    - **Continuous**: Risk Score (0.0 to 1.0).
    - **Multi-class**: 18 Condition categories (MSD pathologies).
    - **Severity**: Low, Moderate, High.

---

## ⚙️ Model Development & Training

The training pipeline follows a rigorous data science workflow:

1.  **Feature Engineering**: Extraction of the **75-feature** biomechanical vector via `feature_extractor.py`.
2.  **Balanced Weights**: Use of class-weighted training to compensate for severe dataset imbalance.
3.  **Temporal Splitting**: 3-fold `TimeSeriesSplit` cross-validation to guarantee zero temporal data leakage.
4.  **Optimization**: Hyperparameter tuning using Bayesian Optimization (Optuna).

---

## 📈 Performance Results (v3.0-Production)

The Ergo Sensor AI achieves state-of-the-art results for real-time ergonomic assessment:

| Metric | Result | Interpretation |
|---|---|---|
| **R² Score (Risk)** | **0.9981** | Near-perfect variance explained in injury risk forecasting. |
| **Accuracy (Cond)** | **99.60%** | Exceptional classification across 18 MSD pathologies. |
| **F1-Score (Severity)** | **0.9411** | Robust generalization on severity despite temporal cross-validation. |
| **F1-Score (Anomaly)** | **0.9906** | Average F1 across 5 distinct per-joint anomaly classifiers. |

---

## 🔍 Explainability with SHAP

One of the most powerful features of Ergo Sensor is **SHAP (SHapley Additive exPlanations)**. 

When a "High Risk" is detected, the AI doesn't just give a number. It uses **TreeExplainer** to calculate the contribution of each of the 75 features. 
- **Example**: "Risk is 85%. **Primary Cause**: Sustained Trunk Flexion (42% contribution) and Shoulder Lag."
- **Benefit**: This allows clinicians to provide specific feedback (e.g., "Adjust your chair height to lower your right shoulder").

---

## 🔄 AI Data Pipeline

1.  **Ingestion**: Raw Quaternions from ESP32 sensors via Firebase.
2.  **Processing**: `angle_math.py` converts orientation to 12 clinical joint angles.
3.  **Extraction**: `feature_extractor.py` computes the dynamic 75-feature time-series vector.
4.  **Inference**: `ai_engine.py` runs the 5-model ensemble.
5.  **Visualization**: Live dashboard, 3D skeleton, and PDF reports with **Anomaly Probability Curves**.
