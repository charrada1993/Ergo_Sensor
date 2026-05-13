# Ergo Sensor v3.0 - Technology Stack

This document outlines the core technologies, libraries, and frameworks used to build the Ergo Sensor AI pipeline and web application, detailing where each piece of technology is applied within the system.

## 1. Backend & Server Infrastructure
The core system is built on a high-performance, asynchronous Python backend designed to handle real-time sensor streams.

*   **Python (3.9+)**: The primary programming language for the entire backend, data processing, and AI pipeline.
*   **Flask**: The lightweight web framework used to serve the frontend dashboard, REST API endpoints, and handle routing (`app.py`).
*   **Flask-SocketIO / eventlet / gevent**: Used for real-time, bi-directional communication between the Python backend and the frontend dashboard. This allows live streaming of 3D skeleton data and AI predictions without page refreshes.
*   **Gunicorn**: The production WSGI HTTP server used to deploy the Flask application.
*   **Firebase Admin SDK**: Used to establish a secure, real-time connection to the Firebase Realtime Database (`firebase_listener.py`) to receive live kinematic data from the IoT sensors.

## 2. Artificial Intelligence & Machine Learning
The "Ergo Sensor AI Engine v3.0-Production" is a multi-model ensemble built to predict musculoskeletal disorder (MSD) risks and detect postural anomalies.

*   **LightGBM**: The core Gradient Boosting framework used for the primary predictive models due to its speed and high accuracy on tabular kinematic data.
    *   *Regressor*: Predicts continuous 10-day risk scores.
    *   *Condition Classifier*: Uses the GBDT/DART booster to classify 18 distinct medical pathologies (e.g., Carpal Tunnel, Lumbar Disc Herniation).
    *   *Severity & Per-Joint Classifiers*: Used for granular anomaly detection per body part.
*   **Scikit-Learn (sklearn)**: Provides the foundational machine learning utilities.
    *   *Isolation Forest*: An unsupervised learning model used for global anomaly detection (detecting irregular, unseen postural movements).
    *   *TimeSeriesSplit*: Used for cross-validation during training to ensure temporal data leakage is prevented.
    *   *Metrics*: Generates confusion matrices, ROC curves, and F1/Accuracy/RMSE scores.
*   **Optuna**: A hyperparameter optimization (HPO) framework used in `retrain_v3.py` to automatically search for the most optimal parameters for the LightGBM models.
*   **SHAP (SHapley Additive exPlanations)**: Used via the `TreeExplainer` to provide explainable AI. It calculates feature importance, revealing exactly *which* joints and movements are driving the AI's risk predictions.
*   **Pandas & NumPy**: The backbone of data manipulation, used extensively in `feature_extractor.py` and `retrain_v3.py` for calculating rolling windows, lags, and acceleration derivatives from the raw time-series data.

## 3. Ergonomic & Biomechanical Processing
Before data reaches the AI, it is processed through validated clinical ergonomic frameworks.

*   **Custom Python Math (`angle_math.py`)**: Uses vector mathematics and trigonometry to calculate 3D joint angles (flexion, extension, deviation) from raw IMU quaternion/Euler data.
*   **RULA (Rapid Upper Limb Assessment)**: Implemented programmatically (`rula_engine.py`) to score upper body posture strain.
*   **REBA (Rapid Entire Body Assessment)**: Implemented programmatically (`reba_engine.py`) to score full-body postural risk.

## 4. Frontend & User Interface
The dashboard is designed to be a "Zero-Dependency" modern web application, favoring vanilla web technologies over heavy frameworks to ensure maximum performance and minimal latency.

*   **HTML5 / Jinja2**: Templates are served by Flask and populated with server-side variables before rendering.
*   **Vanilla CSS3**: Used exclusively for styling. Features advanced CSS variables, glassmorphism (`backdrop-filter`), CSS Grid/Flexbox, and keyframe micro-animations to create a premium, dark-mode clinical interface (`static/style.css`, inline styles in `ai.html`).
*   **Vanilla JavaScript (ES6)**: Handles client-side logic, WebSocket message ingestion, and DOM updates without the overhead of React or Vue.
*   **Socket.IO Client**: Connects to the Flask-SocketIO server to receive live JSON payloads of sensor data and AI predictions.
*   **Three.js / WebGL (via dependencies)**: Used to render the live 3D stick-figure skeleton on the main dashboard (`index.html`).
*   **FontAwesome**: Provides the scalable vector icons used throughout the UI.
*   **Google Fonts**: Uses 'Syne' and 'JetBrains Mono' for modern typography.

## 5. Medical Report Generation
The system generates automated, clinical-grade PDF assessments.

*   **ReportLab (Platypus)**: A robust Python library used in `report_generator.py` to programmatically build complex, multi-page PDF documents with styled tables, headers, footers, and text flow.
*   **Matplotlib**: Used statelessly (`Agg` backend) to generate high-quality PNG charts (learning curves, joint angle trends, ROC curves) which are then embedded directly into the PDF reports and served to the AI dashboard (`generate_eval_plots.py`, `retrain_v3.py`).

## 6. Hardware & Edge Integration (Assumed Context)
While this repository focuses on the software, the software is designed to interface with specific edge hardware.

*   **ESP32 Microcontrollers**: Used to collect data from the physical sensors.
*   **IMU Sensors (BNO085 / MPU6050)**: Provide the raw 9-DOF orientation data.
*   **NVIDIA Jetson Orin / Raspberry Pi**: Edge devices where the Python processing pipeline can be deployed for on-premise computation.

## 7. Cloud & Deployment
*   **Render.com**: The PaaS (Platform as a Service) target for deploying the Flask web application.
*   **Firebase Realtime Database (RTDB)**: Acts as the high-speed message broker between the physical IoT sensors and the deployed Python backend.
