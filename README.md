# 🦾 Ergo Sensor — Musculoskeletal Disorder Risk Assessment System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-black?style=for-the-badge&logo=flask&logoColor=white)
![Firebase](https://img.shields.io/badge/Firebase-Realtime-orange?style=for-the-badge&logo=firebase&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-AI-green?style=for-the-badge)
![Render](https://img.shields.io/badge/Render-Deployment-00d4ff?style=for-the-badge&logo=render&logoColor=white)

**A real-time, AI-powered ergonomic monitoring platform for Musculoskeletal Disorder (MSD) risk assessment using ESP32 IMU sensor networks and Firebase.**

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Software Stack](#-software-stack)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Running the System](#-running-the-system)
- [AI Intelligence](#-ai-intelligence)
- [API Reference](#-api-reference)
- [Deployment](#-deployment)
- [Contributing](#-contributing)

---

## 🔍 Overview

**Ergo Sensor** is a full-stack ergonomic monitoring system that collects real-time orientation data from a network of ESP32-based IMU sensors. It computes joint angles, evaluates ergonomic risk using standardised RULA and REBA scoring methods, and applies a multi-layered AI ensemble (LightGBM + Isolation Forest) to provide a 10-day MSD risk forecast and anomaly detection.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🔴 **Real-Time Monitoring** | Live joint angle and risk data streamed via WebSocket (Socket.IO). |
| 🦾 **38-Feature Vector** | Advanced biomechanical modeling including kinematics and temporal derivatives. |
| 🤖 **AI Risk Forecast** | LightGBM model predicts 10-day musculoskeletal disorder risk probability. |
| 🔍 **Anomaly Probability** | Real-time probability curves for 5 postural disorders (Neck, Shoulder, etc.). |
| 🧠 **SHAP Explainability** | Identifies exactly which joint is the primary driver of risk in real-time. |
| 📐 **RULA / REBA Scoring** | Integrated bilateral ergonomic scoring engines. |
| 📊 **Advanced PDF Reports** | Medical-grade reports with trend charts and clinical AI insights. |
| 🔥 **Firebase Ingestion** | Seamless data sync from ESP32 sensors to the cloud. |

---

## 🏗️ System Architecture

```mermaid
graph TD
    A[ESP32 IMU Sensors] -- "HTTP POST / Firebase" --> B[Cloud Backend (Render)]
    B --> C{Data Processor}
    C --> D[Angle Math Engine]
    C --> E[RULA/REBA Engines]
    C --> F[AI Inference Engine]
    F --> G[10-Day Risk Forecast]
    F --> H[Anomaly Prob Curves]
    F --> I[SHAP Explainability]
    C --> J[Enriched CSV Logger]
    B -- "Socket.IO (Gevent)" --> K[Real-time Dashboard]
    K --> L[Live Gauges]
    K --> M[AI Predictive Analytics]
    K --> N[PDF Report Generator]
```

---

## 💻 Software Stack

| Layer | Technology |
|---|---|
| **Backend** | Python 3.11, Flask 3.0, Flask-SocketIO 5.3 |
| **Real-time** | Socket.IO (Gevent-WebSocket) |
| **WSGI Server** | Gunicorn (Production) / Werkzeug (Dev) |
| **Database** | Firebase Realtime Database |
| **AI / ML** | LightGBM, scikit-learn, SHAP, Joblib |
| **Data** | Pandas, NumPy |
| **Reports** | ReportLab, Matplotlib |

---

## 📂 Project Structure

```
MSD_System/
│
├── app.py                    # Flask application & Socket.IO routes
├── config.py                 # Central config (Firebase, Port, Sensors)
├── data_processor.py         # Main pipeline (Angles -> RULA/REBA -> AI)
├── ai_engine.py              # AI Inference (LightGBM, SHAP, Anomaly)
├── firebase_listener.py      # Real-time Firebase stream listener
├── angle_math.py             # IMU Quaternions -> Joint Angles
├── feature_extractor.py      # 38-feature vector generation
├── csv_logger.py             # Enriched CSV logging (Dataset ready)
├── report_generator.py       # PDF generator with Anomaly Curves
│
├── models/                   # AI Model binaries (.txt, .pkl)
├── static/                   # Dashboard CSS/JS assets
├── templates/                # HTML Templates (Jinja2)
│
├── AI.md                     # [NEW] Detailed AI Documentation
├── READY_TO_DEPLOY.md        # [NEW] Render Deployment Guide
└── README.md                 # Project Overview
```

---

## ⚙️ Configuration

Edit `config.py` for your environment:

```python
class Config:
    # Firebase settings
    FIREBASE_DATABASE_URL = 'https://your-project.firebaseio.com/'
    
    # Credentials (Local only)
    # On Render, use FIREBASE_CREDS_JSON env variable!
    FIREBASE_CREDENTIALS_PATH = 'firebase-key.json'

    # Expected Sensor IDs
    EXPECTED_SENSORS = ['NECK', 'UPPER_BACK', 'R_BICEPS', ...]
```

---

## ▶️ Running the System

### 🛠️ Development (Local)
```bash
python app.py
```
*Port defaults to 5000. Debug mode enabled.*

### 🚀 Production (Render)
The system uses **Gunicorn** with the **Gevent** worker for high-concurrency WebSockets:
```bash
gunicorn -k geventwebsocket.gunicorn.workers.GeventWebSocketWorker -w 1 app:app
```

---

## 🤖 AI Intelligence

Ergo Sensor uses a 38-feature biomechanical model to track ergonomic risk. 

**Detailed technical explanation of the AI logic is available in [AI.md](AI.md).**

Key components:
- **LightGBM Regressor**: Cumulative 10-day risk forecasting.
- **Isolation Forest**: Instant global anomaly detection.
- **Granular Classifiers**: Real-time probability curves for 5 postural disorders.
- **SHAP Engine**: Root cause joint analysis.

---

## 🌐 API Reference

| Method | Route | Description |
|---|---|---|
| `POST` | `/api/data` | Receive IMU data from ESP32 |
| `POST` | `/api/calibrate` | Set current pose as zero reference |
| `GET` | `/api/sensors` | Return online/offline status of all sensors |
| `GET` | `/api/csv/latest` | Download the latest enriched CSV session |
| `POST` | `/api/report/generate`| Generate PDF report with anomaly curves |

---

## 🌍 Deployment

The system is optimized for **Render.com**. 

### Critical Deployment Steps:
1.  **Python Version**: Set `PYTHON_VERSION` to `3.11.0` in Render environment settings.
2.  **Firebase Security**: Add the full content of your Firebase JSON to the `FIREBASE_CREDS_JSON` environment variable.
3.  **Start Command**: `gunicorn -k geventwebsocket.gunicorn.workers.GeventWebSocketWorker -w 1 app:app`

See [READY_TO_DEPLOY.md](READY_TO_DEPLOY.md) for the full walkthrough.

---

## 🤝 Contributing

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/new-stuff`).
3. Commit your changes.
4. Open a Pull Request.

---

## 📜 License

This project is licensed under the **MIT License**.

---

**Charrada** — [GitHub](https://github.com/charrada1993)
