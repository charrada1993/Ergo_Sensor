# 🧠 Ergo Sensor — AI Intelligence & Biomechanical Modeling

This document provides a deep dive into the Artificial Intelligence core of the **Ergo Sensor** system. It explains the scientific rationale, the model architectures, and the data pipeline used to prevent Musculoskeletal Disorders (MSD).

---

## 🔬 Why This Model Choice?

We selected an **Ensemble Learning** approach based on **LightGBM** and **Isolation Forest** for several technical reasons:

3.  **DART Boosting**: Ergo Sensor v3.0 uses the **DART** (Dropouts meet Multiple Additive Regression Trees) booster for condition classification, preventing over-fitting on dominant classes and improving recall on rare pathologies.
4.  **Handling Imbalance**: Postural anomalies are rare. LightGBM's `is_unbalance`, `scale_pos_weight`, and class-balanced training allow us to detect rare injuries effectively.
5.  **Unsupervised Hybrid**: By combining Isolation Forest (Unsupervised) with LightGBM (Supervised), we can detect both *known* disorders (like Neck Hyperflexion) and *unknown* dangerous movements.

---

## 📊 The Dataset: `dataset_TMS_enriched.csv`

The model was trained on a high-quality dataset specifically engineered for Musculoskeletal Disorder (MSD) research:

- **Volume**: ~50,000+ data points collected at 10Hz.
- **Input Features (59)**: 
    - 12 raw joint angles (Clinical standard).
    - 19 temporal statistics (Moving averages/variances).
    - 7 dynamic features (Angular velocities).
    - 5 bilateral asymmetry deltas.
    - 7 energy proxies (Velocity × Duration).
    - 2 composite load scores.
    - 7 raw angle overlays and posture flags.
- **Target Classes**: 
    - **Continuous**: Risk Score (0.0 to 1.0).
    - **Multi-class**: 18 Condition categories (MSD pathologies).
    - **Severity**: Low, Moderate, High.

---

## ⚙️ Model Development & Training

The training pipeline follows a rigorous data science workflow:

1.  **Feature Engineering**: Extraction of the **59-feature** biomechanical vector via `retrain_v3.py`.
2.  **Balanced Weights**: Use of class-weighted training and balanced sample weights to compensate for severe dataset imbalance (some pathologies have <10 samples).
3.  **Temporal Splitting**: 80/20 temporal split (non-shuffled) to ensure the model can predict future states from past sequences.
4.  **Optimization**: Hyperparameter tuning using Bayesian Optimization (Optuna) to maximize the AUC-ROC curve.

---

## 📈 Performance Results

The Ergo Sensor AI achieves state-of-the-art results for real-time ergonomic assessment:

| Metric | Result | Interpretation |
|---|---|---|
| **R² Score (Risk)** | **0.9966** | Extremely high variance explained in injury risk forecasting. |
| **Accuracy (Cond)** | **99.52%** | Near-perfect classification across 18 MSD pathologies. |
| **F1-Score (Macro)** | **0.9661** | Excellent balance even on severely imbalanced classes. |
| **Inference Time** | **< 0.5ms** | Instantaneous feedback at high sampling rates. |

---

## 🔍 Explainability with SHAP

One of the most powerful features of Ergo Sensor is **SHAP (SHapley Additive exPlanations)**. 

When a "High Risk" is detected, the AI doesn't just give a number. It uses **TreeExplainer** to calculate the contribution of each joint. 
- **Example**: "Risk is 85%. **Primary Cause**: Right Shoulder Abduction (42% contribution)."
- **Benefit**: This allows clinicians to provide specific feedback (e.g., "Adjust your chair height to lower your right shoulder").

---

## 🔄 AI Data Pipeline

1.  **Ingestion**: Raw Roll/Pitch/Yaw from ESP32 sensors via Firebase.
2.  **Processing**: `angle_math.py` converts orientation to 12 clinical joint angles.
3.  **Extraction**: `feature_extractor.py` computes the 38-feature vector.
4.  **Inference**: `ai_engine.py` runs the models.
5.  **Visualization**: Live dashboard gauges and PDF reports with time-series **Anomaly Probability Curves**.
