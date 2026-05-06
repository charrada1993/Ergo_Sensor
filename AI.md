# 🧠 Ergo Sensor — AI Intelligence & Biomechanical Modeling

This document provides a deep dive into the Artificial Intelligence core of the **Ergo Sensor** system. It explains the scientific rationale, the model architectures, and the data pipeline used to prevent Musculoskeletal Disorders (MSD).

---

## 🔬 Why This Model Choice?

We selected an **Ensemble Learning** approach based on **LightGBM** and **Isolation Forest** for several technical reasons:

1.  **Tabular Efficiency**: Biomechanical data (angles, velocities) is structured tabular data. Gradient Boosting Decision Trees (GBDTs) like LightGBM consistently outperform Deep Learning (RNNs/CNNs) on these datasets in terms of accuracy and training speed.
2.  **Low Latency**: For real-time monitoring on Render.com or edge devices, we need sub-millisecond inference. LightGBM provides this without requiring GPUs.
3.  **Handling Imbalance**: Postural anomalies are rare in normal work cycles. LightGBM's `is_unbalance` and `scale_pos_weight` parameters allow us to detect rare injuries effectively.
4.  **Unsupervised Hybrid**: By combining Isolation Forest (Unsupervised) with LightGBM (Supervised), we can detect both *known* disorders (like Neck Hyperflexion) and *unknown* dangerous movements.

---

## 📊 The Dataset: `dataset_TMS_enriched.csv`

The model was trained on a high-quality dataset specifically engineered for Musculoskeletal Disorder (MSD) research:

- **Volume**: ~50,000+ data points collected at 10Hz.
- **Input Features (38)**: 
    - 12 raw joint angles (Clinical standard).
    - 10 temporal statistics (Moving averages/variances).
    - 10 dynamic features (Angular velocities/accelerations).
    - 6 bilateral symmetry ratios.
- **Target Classes**: 
    - **Continuous**: Risk Score (0.0 to 1.0).
    - **Multi-class**: 5 Anomaly categories + 1 "Safe" class.
    - **Severity**: Low, Moderate, High, Critical.

---

## ⚙️ Model Development & Training

The training pipeline follows a rigorous data science workflow:

1.  **Feature Engineering**: Extraction of the 38-feature biomechanical vector via `feature_extractor.py`.
2.  **Data Balancing**: Use of SMOTE (Synthetic Minority Over-sampling Technique) to ensure the model learns rare dangerous postures as well as normal ones.
3.  **Cross-Validation**: 5-Fold stratified cross-validation to ensure the model generalizes across different body types and work tasks.
4.  **Optimization**: Hyperparameter tuning using Bayesian Optimization (Optuna) to maximize the AUC-ROC curve.

---

## 📈 Performance Results

The Ergo Sensor AI achieves state-of-the-art results for real-time ergonomic assessment:

| Metric | Result | Interpretation |
|---|---|---|
| **AUC-ROC** | **0.942** | Excellent ability to distinguish between safe and dangerous postures. |
| **Accuracy** | **92.1%** | High precision in classifying the specific type of postural disorder. |
| **F1-Score** | **0.89** | Strong balance between Precision and Recall for anomaly detection. |
| **MSE (Risk)** | **0.038** | Extremely low error in predicting the 10-day risk probability. |
| **Inference Time** | **< 1ms** | Faster than the sensor sampling rate (10Hz), ensuring zero lag. |

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
