# Ergo Sensor v3.0 - Project Structure

This document outlines the global structure of the Ergo Sensor repository, explaining the purpose of each major component, directory, and critical file in the system.

## Global Architecture Overview
The Ergo Sensor system is designed as an edge-to-cloud IoT pipeline. It ingests high-frequency kinematic data from hardware sensors, processes it through clinical ergonomic frameworks (RULA/REBA) and a machine learning AI Engine, and visualizes the risk metrics in a real-time web dashboard. Finally, it generates automated clinical PDF reports.

## Directory Structure

```text
c:\MSD_System\
├── .git/                      # Version control
├── csv_data/                  # Local storage for recorded kinematic sessions
├── dist/                      # Compiled frontend assets / production bundles
├── Ergo_Sensor_Project_Report/# Academic/Project documentation assets
├── logs/                      # System operation logs
├── models/                    # Serialized AI models, Metadata, and Evaluation Plots
├── PFE_LaTeX_Template/        # LaTeX source code for the final university report
├── plots/                     # Static images served to the Web Dashboard
├── reports/                   # Generated clinical PDF reports
├── static/                    # Frontend static assets (CSS, JS, Fonts, Images)
├── templates/                 # Frontend HTML templates (Jinja2)
└── (Root Python Scripts)      # Core backend logic
```

## Core Modules & Files

### 1. Web Application & Routing
These files handle the web server, HTTP routing, and real-time WebSocket communication.
*   **`app.py`**: The main entry point for the Flask web application. Initializes the server, defines API routes (`/api/ai-metrics`), serves HTML pages, and manages the Socket.IO lifecycle.
*   **`socket_manager.py`**: Handles incoming WebSocket connections and broadcasts data to connected web clients.

### 2. Frontend (UI/UX)
Located in `templates/` and `static/`.
*   **`templates/index.html`**: The main real-time dashboard displaying the 3D skeleton and live RULA/REBA scores.
*   **`templates/ai.html`**: The AI Predictive Analytics dashboard displaying risk forecasts, anomaly detection, and comprehensive model evaluation plots.
*   **`templates/reports.html`**: Interface for viewing and downloading generated PDF reports.
*   **`templates/csv_view.html`**: Interface for browsing historical CSV session data.
*   **`static/style.css`**: The master stylesheet defining the premium, dark-mode clinical aesthetic, animations, and responsive layouts.

### 3. Data Processing & Ergonomic Engines
These scripts process raw incoming data and apply clinical formulas.
*   **`data_processor.py`**: The central coordinator that receives raw sensor data, triggers angle calculations, and orchestrates the RULA/REBA scoring.
*   **`angle_math.py`**: Contains the complex vector mathematics to convert raw sensor Quaternions/Euler angles into standard biomechanical joint angles (flexion, extension, etc.).
*   **`rula_engine.py` / `reba_engine.py`**: Programmatic implementations of the Rapid Upper Limb Assessment and Rapid Entire Body Assessment clinical risk scoring frameworks.
*   **`rula_ref.py` / `reba_ref.py`**: Reference tables and constants used by the ergonomic engines.

### 4. AI Engine v3.0-Production
The predictive core of the system.
*   **`ai_engine.py`**: The runtime inference engine. Loads the trained `.txt` and `.pkl` models and executes live predictions (Risk Score, Condition, Severity, Anomalies) on incoming data streams.
*   **`feature_extractor.py`**: Prepares raw data for the AI. Generates the 75-feature vector by calculating 15-frame rolling means/standard deviations, lag features, and velocity/acceleration derivatives to provide temporal context to the models.
*   **`retrain_v3.py`**: The highly optimized training pipeline. Handles data loading, Optuna hyperparameter tuning, TimeSeriesSplit cross-validation, LightGBM model training, SHAP feature importance analysis, and automatic plotting.
*   **`retrain_scratch.py` / `retrain_improved.py`**: Legacy/experimental training scripts.
*   **`generate_eval_plots.py`**: Standalone script to generate the 10-plot evaluation suite (ROC, PR, Confusion Matrices) without retraining models.

### 5. Reporting & Logging
*   **`report_generator.py`**: Uses ReportLab to generate highly detailed, clinical-grade A4 PDF documents. It embeds charts, RULA/REBA summaries, and AI Insights.
*   **`csv_logger.py`**: Handles safely writing live, high-frequency sensor streams to local CSV files in the `csv_data/` directory for permanent storage and later retraining.

### 6. IoT & Database Integration
*   **`firebase_listener.py`**: Connects to the Firebase Realtime Database. Listens for new kinematic data pushed by the ESP32 hardware and pipes it into the Python `data_processor.py`.

### 7. Configuration & Utilities
*   **`config.py`**: Centralized configuration file storing database keys, model paths, and system parameters.
*   **`requirements.txt`**: Defines the Python package dependencies required to run the backend.
*   **`msd-monitor-system-firebase...json`**: The private service account key used to authenticate with Firebase.
*   **`condition_mappings.json`**: Maps integer labels used by the AI classifier back to human-readable medical conditions (e.g., `0 -> normal`, `1 -> carpal_tunnel`).

## The Data Flow
1.  **Hardware:** IMU sensors send data to an ESP32, which pushes it to Firebase RTDB.
2.  **Ingestion:** `firebase_listener.py` detects the new data and sends it to `data_processor.py`.
3.  **Processing:** `angle_math.py` calculates joints, then `rula_engine.py` and `reba_engine.py` calculate immediate ergonomic risk.
4.  **AI Inference:** The data is passed to `feature_extractor.py` to create the 75-feature window, which is then fed to `ai_engine.py` to predict long-term risk and anomalies.
5.  **Broadcast:** The complete payload (angles, RULA/REBA, AI predictions) is broadcast via `socket_manager.py`.
6.  **Visualization:** The web browser receives the payload and updates the 3D skeleton and charts in `index.html` and `ai.html`.
7.  **Reporting (Optional):** When a session ends, `report_generator.py` analyzes the saved CSV and generates a PDF report.
