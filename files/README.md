# ⚠️ Legacy AI Reference Scripts (v1.0)

This directory contains the original, legacy v1.0 reference scripts, documentation, and source code of the initial AI prototype.

> [!WARNING]
> The scripts in this directory are kept **solely for historical reference and backward compatibility**. They use a legacy 14-feature snapshot model and basic neural network architectures (CNN 1D, LSTM, and Hybrid models).
>
> For the active, production-grade codebase (v3.0-Production), refer to the files in the **root directory** of the repository.

---

## 📂 Legacy File Manifest
*   **`01_generate_dataset.py`**: Legacy script for generating synthetic, 14-feature biomechanical datasets.
*   **`02_train_models.py`**: Legacy script for training five prototype neural network models (CNN, LSTM, Hybrid CNN+LGB) using basic random splitting.
*   **`03_inference.py`**: Legacy single-inference script for checking prototype predictions.
*   **`run_complete_pipeline.py`**: Legacy coordinator script to run v1.0 end-to-end dataset generation and training.
*   **`TECHNICAL_SUMMARY.txt`**: Legacy documentation notes describing the v1.0 neural network architectures.

---

## 🔄 Differences: Legacy (v1.0) vs. Active Production (v3.0)

| Feature | Legacy v1.0 (`files/`) | Active Production v3.0 (Root `c:/MSD_System/`) |
|---|---|---|
| **Features Used** | **14 base features** (static postural snapshots) | **75 engineered features** (Rolling statistics, lag indices, bilateral asymmetries, velocities, composite joint loads, and accelerations) |
| **Model Architectures** | CNN 1D, LSTM, Hybrid (CNN+LGB), LightGBM | Highly optimized, multi-model ensemble of LightGBM Regressors, Condition classifiers, Severity classifiers, and 5 binary joint anomalies |
| **Cross-Validation** | Standard Random Split (suffers from temporal data leakage) | **`TimeSeriesSplit` (3-fold)** (ensures temporal validity and prevents data leakage) |
| **Hyperparameters** | Hardcoded/manual parameters | **Optuna Optimization** (automated 10-trial hyperparameter searches) |
| **Ergonomic Engines** | Simple angles | Programmatic bilateral **RULA & REBA Engines** with dynamic calibration |
| **Explainability** | Basic SHAP plots | Native **SHAP TreeExplainer** fully integrated into the live clinician dashboard |
| **Visualizations** | None | Real-time **Three.js 3D Digital Twin** skeleton mirror |

---

## 🚀 How to Run the Current Production Pipeline

To train, optimize, and serve the active v3.0-Production system, use the following files located in the root directory:
1.  **Model Training**: Run [retrain_v3.py](file:///c:/MSD_System/retrain_v3.py) to execute the Optuna HPO, TimeSeriesSplit CV, and feature engineering.
2.  **Telemetry Processing**: Check [angle_math.py](file:///c:/MSD_System/angle_math.py) for the quaternion-to-joint conversions and [feature_extractor.py](file:///c:/MSD_System/feature_extractor.py) for the 75-feature window extraction.
3.  **Real-Time Dashboard & Backend Server**: Start the Flask app by executing [app.py](file:///c:/MSD_System/app.py).
