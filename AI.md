# 🧠 Ergo Sensor — AI Intelligence & Biomechanical Modeling

This document provides a deep dive into the Artificial Intelligence core of the **Ergo Sensor** system. It explains the scientific rationale, the model architectures, and the data pipeline used to prevent Musculoskeletal Disorders (MSDs).

---

## 🔬 Why AI for Ergonomics?

Traditional ergonomic assessments (like RULA or REBA) are **reactive** and **static**:
1. They capture a single moment in time.
2. they rely on manual observation.
3. They don't account for cumulative fatigue or complex joint interactions.

**Ergo Sensor AI** transforms this into a **predictive** and **dynamic** process by:
- **Continuous Monitoring**: Analyzing 10 frames per second.
- **Pattern Recognition**: Identifying subtle postural "signatures" that lead to injury.
- **Forecasting**: Predicting the probability of risk over a 10-day horizon.

---

## 📐 The Biomechanical Feature Vector (38 Features)

Instead of just looking at raw angles, our AI models ingest a high-dimensional vector representing the worker's full-body kinematics:

| Category | Count | Description |
|---|---|---|
| **Direct Angles** | 12 | Flexion/Extension, Abduction, and Rotation for Neck, Trunk, and Limbs. |
| **Bilateral Symmetry** | 6 | Differences between Left and Right sides (e.g., Shoulder imbalance). |
| **Temporal Stats** | 10 | Mean, Standard Deviation, and 95th Percentile over a 60-second sliding window. |
| **Motion Dynamics** | 10 | Angular velocities and accelerations to detect sudden or repetitive stress. |

---

## 🤖 The Model Ensemble

Ergo Sensor uses a multi-layered AI approach:

### 1. Risk Forecasting (LightGBM Regressor)
*   **Goal**: Predict the "Cumulative Damage" probability over 10 days.
*   **How it works**: It analyzes the 60-second window of movement patterns. If the model sees high-frequency micro-movements combined with extreme angles, the risk score rises.
*   **Why LightGBM?**: It is extremely fast for real-time inference and handles tabular biomechanical data better than deep learning in low-latency environments.

### 2. Anomaly Detection (Isolation Forest)
*   **Goal**: Detect "Unseen" or "Dangerous" movements instantly.
*   **How it works**: This is an unsupervised model. It learns the "normal" workspace movements of a worker. Anything outside this (e.g., a sudden fall or an extreme twist) is flagged as a high anomaly score.

### 3. Postural Disorder Classifiers (Granular Anomaly Models)
We use 5 dedicated LightGBM classifiers to identify specific clinical conditions:
- **Neck Hyperflexion**: Excessive forward head tilt.
- **Shoulder Overextension**: Reaching too far or too high.
- **Wrist Strain**: Repetitive or extreme wrist deviation.
- **Trunk Torsion**: Dangerous twisting of the spine.
- **Elbow Hyperextension**: Locking joints under load.

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
3.  **Extraction**: `feature_extractor.py` computes the 38-feature vector (stats + dynamics).
4.  **Inference**: `ai_engine.py` runs the models (Risk, Anomaly, Classifiers).
5.  **Visualization**:
    *   **Dashboard**: Live gauges and SHAP alerts.
    *   **PDF Reports**: Time-series **Probability Curves** showing exactly when and why an anomaly occurred.

---

## 🛠️ Model Training

The models are trained using the `dataset_TMS_enriched.csv` collected during real-world work sessions.
- **Algorithm**: LightGBM (Gradient Boosting Decision Tree).
- **Validation**: 5-fold cross-validation focusing on Precision/Recall for anomaly detection.
- **Serialization**: Models are saved as `.txt` (LGBM) and `.pkl` (Joblib) in the `models/` directory.
