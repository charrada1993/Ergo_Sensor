# 🦾 Ergo Sensor — Musculoskeletal Disorder Risk Assessment System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-black?style=for-the-badge&logo=flask&logoColor=white)
![Firebase](https://img.shields.io/badge/Firebase-Realtime-orange?style=for-the-badge&logo=firebase&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-AI-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)

**A real-time, AI-powered ergonomic monitoring platform for Musculoskeletal Disorder (MSD) risk assessment using ESP32 IMU sensor networks and Firebase.**

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Hardware Requirements](#-hardware-requirements)
- [Software Stack](#-software-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Running the System](#-running-the-system)
- [ESP32 Sensor Setup](#-esp32-sensor-setup)
- [API Reference](#-api-reference)
- [Dashboard Pages](#-dashboard-pages)
- [AI Models](#-ai-models)
- [Ergonomic Scoring](#-ergonomic-scoring-rula--reba)
- [Report Generation](#-report-generation)
- [User Roles & Authentication](#-user-roles--authentication)
- [Testing](#-testing)
- [Deployment](#-deployment)

---

## 🔍 Overview

**Ergo Sensor** is a full-stack ergonomic monitoring system that collects real-time orientation data from a network of ESP32-based IMU sensors placed on a worker's body, computes joint angles, evaluates ergonomic risk using standardised RULA and REBA scoring methods, and applies machine learning models (LightGBM + Isolation Forest) to provide a 10-day MSD risk forecast and anomaly detection.

Data is streamed from ESP32 devices → Firebase Realtime Database → Flask backend → WebSocket dashboard — all in real time.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🔴 **Real-Time Monitoring** | Live joint angle and risk data streamed via WebSocket (Socket.IO) |
| 🤖 **AI Risk Forecast** | LightGBM model predicts 10-day musculoskeletal disorder risk probability |
| 🔍 **Anomaly Detection** | Isolation Forest + multi-class LightGBM anomaly classifiers |
| 🧠 **SHAP Explainability** | TreeExplainer identifies which joints drive the risk prediction |
| 📐 **RULA / REBA Scoring** | Full bilateral (left/right) RULA and REBA ergonomic scoring engines |
| 📊 **PDF Reports** | Medical-grade, auto-generated PDF reports with trend charts and AI insights |
| 🔥 **Firebase Integration** | Dual ingestion: direct ESP32 HTTP POST + Firebase Realtime Database stream |
| 📁 **CSV Logging** | Continuous session logging with per-frame joint angles and risk scores |
| 🌐 **Role-Based Access** | Separate Doctor and Patient dashboards with session authentication |
| 🌙 **Dark / Light Theme** | Glassmorphic UI with instant theme switching |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ESP32 Sensor Network                         │
│  [NECK] [UPPER_BACK] [R/L_BICEPS] [R/L_FOREARM] [R/L_HAND]        │
│  [R/L_THIGH] [R/L_SHANK]  → IMU (Roll / Pitch / Yaw @ 10 Hz)      │
└───────────────────────┬─────────────────────────┬───────────────────┘
                        │ HTTP POST                │ Firebase RTDB
                        ▼                          ▼
              ┌──────────────────┐     ┌──────────────────────┐
              │  /api/data route │     │  FirebaseListener     │
              │  (Direct ingest) │     │  (Background thread)  │
              └────────┬─────────┘     └──────────┬───────────┘
                       └──────────┬───────────────┘
                                  ▼
                      ┌───────────────────────┐
                      │    DataProcessor       │
                      │  • Joint Angle Math    │
                      │  • RULA Engine         │
                      │  • REBA Engine         │
                      │  • Risk Engine         │
                      │  • Feature Extractor   │
                      │  • AI Prediction       │
                      │  • CSV Logger          │
                      └───────────┬───────────┘
                                  │ Socket.IO emit
                                  ▼
                      ┌───────────────────────┐
                      │   Flask + Socket.IO    │
                      │   Web Dashboard        │
                      │  • Live Gauges         │
                      │  • RULA/REBA Charts    │
                      │  • AI Predictions      │
                      │  • PDF Reports         │
                      └───────────────────────┘
```

---

## 🔧 Hardware Requirements

| Component | Details |
|---|---|
| **Microcontrollers** | ESP32 (any variant with Wi-Fi) |
| **IMU Sensors** | MPU-6050, MPU-9250, or BNO055 (SPI/I2C) |
| **Sensor Count** | Up to 12 sensors (bilateral full-body) |
| **Server Machine** | Any PC/Raspberry Pi on the same LAN |
| **Network** | Wi-Fi 2.4 GHz (ESP32 ↔ Flask server) |

### Recommended Sensor Placement

```
NECK           →  Cervical spine
UPPER_BACK     →  Thoracic spine / trunk
R_BICEPS       →  Right upper arm
L_BICEPS       →  Left upper arm
R_FOREARM      →  Right forearm
L_FOREARM      →  Left forearm
R_HAND         →  Right wrist/hand
L_HAND         →  Left wrist/hand
R_THIGH        →  Right thigh
L_THIGH        →  Left thigh
R_SHANK        →  Right lower leg
L_SHANK        →  Left lower leg
```

---

## 💻 Software Stack

| Layer | Technology |
|---|---|
| **Backend** | Python 3.10+, Flask 3.0, Flask-SocketIO 5.3 |
| **Real-time** | Socket.IO (WebSocket) |
| **Database** | Firebase Realtime Database (Google Cloud) |
| **AI / ML** | LightGBM, scikit-learn (Isolation Forest), SHAP |
| **Data** | NumPy, Pandas |
| **Reports** | ReportLab (PDF generation), Matplotlib |
| **Frontend** | Vanilla HTML5 / CSS3 / JavaScript, Font Awesome 6 |
| **Firmware** | C++ / Arduino (ESP32) |

---

## 📂 Project Structure

```
MSD_System/
│
├── app.py                    # Flask application entry point & all routes
├── config.py                 # Centralised configuration (sensors, paths, risk params)
├── data_processor.py         # Core pipeline: angles → RULA/REBA → AI → emit
├── ai_engine.py              # LightGBM + Isolation Forest + SHAP inference
├── firebase_listener.py      # Firebase RTDB real-time stream listener
├── angle_math.py             # Joint angle computation from raw IMU quaternions
├── risk_engine.py            # Global composite risk score computation
├── rula_engine.py            # Full bilateral RULA scoring engine
├── reba_engine.py            # Full bilateral REBA scoring engine
├── feature_extractor.py      # Sliding-window feature extraction for AI models
├── csv_logger.py             # Continuous per-frame CSV session logging
├── report_generator.py       # ReportLab PDF report builder
├── socket_manager.py         # Socket.IO instance and event registration
├── requirements.txt          # Python dependencies
│
├── models/                   # Trained AI model files (not committed to git)
│   ├── lgbm_risk_10d.txt     # LightGBM 10-day risk forecast model
│   ├── isolation_forest.pkl  # Isolation Forest anomaly detector
│   ├── scaler_if.pkl         # StandardScaler for Isolation Forest features
│   ├── lgbm_<anomaly>.txt    # 5× anomaly classification models
│   └── model_metadata.json   # Feature columns, sequence length, anomaly names
│
├── templates/                # Jinja2 HTML templates
│   ├── base.html             # Shared navbar, theme toggle, scripts
│   ├── index.html            # Main live dashboard
│   ├── system.html           # System status & calibration
│   ├── ai.html               # AI predictive analytics page
│   ├── rula.html             # RULA ergonomic assessment viewer
│   ├── reba.html             # REBA ergonomic assessment viewer
│   ├── csv_view.html         # CSV session file browser
│   ├── reports.html          # PDF report list & download
│   ├── login.html            # Login page
│   └── sensors.html          # ESP32 sensor status table
│
├── static/
│   ├── style.css             # Global dark-mode glassmorphic stylesheet
│   ├── dashboard.js          # Live dashboard Socket.IO client logic
│   ├── rula.js               # RULA page client
│   └── reba.js               # REBA page client
│
├── csv_data/                 # Auto-created: session CSV logs
├── reports/                  # Auto-created: generated PDF reports
├── logs/                     # Auto-created: application logs
│
├── stress_test.py            # HTTP load testing tool
├── test_backend.py           # Backend unit tests
├── test_logic.py             # Core logic unit tests
└── .gitignore
```

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/charrada1993/MSD_System.git
cd MSD_System
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Additional packages required (not listed in base requirements.txt):

```bash
pip install firebase-admin pandas matplotlib scikit-learn lightgbm shap joblib
```

### 4. Add Firebase credentials

Place your Firebase Admin SDK JSON key file in the project root:

```
MSD_System/
└── msd-monitor-system-firebase-adminsdk-fbsvc-XXXXXXXXXX.json
```

> Download from: **Firebase Console → Project Settings → Service Accounts → Generate New Private Key**

---

## ⚙️ Configuration

Edit `config.py` to match your environment:

```python
class Config:
    # Firebase
    FIREBASE_DATABASE_URL = 'https://YOUR-PROJECT-default-rtdb.region.firebasedatabase.app/'
    FIREBASE_CREDENTIALS_PATH = 'your-firebase-key.json'

    # Sensor IDs — must match what your ESP32 devices send as `sensor_id`
    EXPECTED_SENSORS = [
        'NECK', 'UPPER_BACK',
        'R_BICEPS', 'L_BICEPS',
        'R_FOREARM', 'L_FOREARM',
        'R_HAND', 'L_HAND',
        'R_THIGH', 'L_THIGH',
        'R_SHANK', 'L_SHANK'
    ]

    # Data posting rate from ESP32 (milliseconds)
    POST_INTERVAL_MS = 100   # 10 Hz
```

---

## ▶️ Running the System

```bash
python app.py
```

The server will start and print:

```
=======================================
   ERGO SENSOR SERVER STARTED WITH AI
=======================================
Local: http://127.0.0.1:5000
LAN:   http://192.168.x.x:5000
=======================================
```

Open your browser at `http://127.0.0.1:5000` and log in.

---

## 📡 ESP32 Sensor Setup

Each ESP32 must POST IMU data to the Flask server at `/api/data` every 100 ms:

### HTTP POST Payload

```json
POST http://192.168.x.x:5000/api/data
Content-Type: application/json

{
  "sensor_id": "NECK",
  "roll": 1.23,
  "pitch": -4.56,
  "yaw": 0.78,
  "timestamp": 1714400000
}
```

### Arduino Example (ESP32)

```cpp
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

const char* ssid     = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";
const char* server   = "http://192.168.x.x:5000/api/data";
const char* sensorId = "NECK";   // change per device

void sendData(float roll, float pitch, float yaw) {
    HTTPClient http;
    http.begin(server);
    http.addHeader("Content-Type", "application/json");

    StaticJsonDocument<200> doc;
    doc["sensor_id"] = sensorId;
    doc["roll"]      = roll;
    doc["pitch"]     = pitch;
    doc["yaw"]       = yaw;
    doc["timestamp"] = millis() / 1000;

    String body;
    serializeJson(doc, body);
    http.POST(body);
    http.end();
}
```

### Firebase Alternative

ESP32 can also write directly to Firebase RTDB at path `/sensor_data/<SENSOR_ID>` — the `FirebaseListener` will pick it up automatically.

---

## 🌐 API Reference

### Public (no auth)

| Method | Route | Description |
|---|---|---|
| `POST` | `/api/data` | Receive IMU data from ESP32 |
| `GET` | `/api/time` | Return current Unix timestamp |
| `GET` | `/api/sensors` | Return online/offline status of all sensors |

### Doctor-only (authenticated)

| Method | Route | Description |
|---|---|---|
| `POST` | `/api/calibrate` | Set current orientations as zero reference |
| `POST` | `/api/predict` | Run AI prediction on a feature dict |
| `GET` | `/api/csv/list` | List all saved CSV session files |
| `GET` | `/api/csv/download/<file>` | Download a CSV file |
| `DELETE` | `/api/csv/delete/<file>` | Delete a CSV session file |
| `GET` | `/api/csv/latest` | Download the latest CSV file |
| `POST` | `/api/report/generate` | Generate PDF from latest CSV |
| `GET` | `/api/reports/list` | List all generated PDF reports |
| `GET` | `/api/reports/download/<file>` | Download a PDF report |

### WebSocket Events (Socket.IO)

| Event | Direction | Payload |
|---|---|---|
| `angles` | Server → Client | Joint angles, RULA, REBA, risk, AI predictions |
| `raw_sensors` | Server → Client | Raw roll/pitch/yaw per sensor ID |

---

## 🖥️ Dashboard Pages

| Route | Role | Description |
|---|---|---|
| `/` | All | **Live Dashboard** — real-time joint gauges, RULA/REBA scores, AI predictions |
| `/system` | All | **System Status** — sensor connectivity, calibration control |
| `/ai` | Doctor | **AI Analytics** — LightGBM risk forecast, anomaly scores, SHAP insights |
| `/rula` | Doctor | **RULA Viewer** — bilateral RULA breakdown per body region |
| `/reba` | Doctor | **REBA Viewer** — bilateral REBA breakdown per body region |
| `/csv-view` | Doctor | **CSV Data** — browse, download, or delete session logs |
| `/reports` | Doctor | **Reports** — generate and download PDF risk reports |
| `/sensors` | All | **Sensor Status** — ESP32 device online/offline table |

---

## 🤖 AI Models

The AI engine (`ai_engine.py`) loads 3 types of models from the `models/` directory:

### 1. LightGBM Risk Forecast (`lgbm_risk_10d.txt`)
- **Input:** 60-frame sliding window of joint angle statistics
- **Output:** Probability (0.0–1.0) of MSD risk exceedance over the next 10 days
- **Risk Levels:** `SAFE` (<0.4) | `LOW` (<0.6) | `MODERATE` (<0.8) | `HIGH` (≥0.8)

### 2. Isolation Forest (`isolation_forest.pkl`)
- **Input:** Current frame feature vector (scaled)
- **Output:** Anomaly score (0.0–1.0). Values >0.6 indicate unusual kinematics.

### 3. Anomaly Classifiers (5× `lgbm_<name>.txt`)
- **Input:** Same feature vector as risk model
- **Output:** Per-class probability for 5 postural anomaly categories
- Identified by SHAP TreeExplainer for root-cause joint analysis

### Model Metadata (`model_metadata.json`)
```json
{
  "feature_cols": ["neck_mean", "shoulder_p95", ...],
  "if_features": ["..."],
  "seq_len": 60,
  "anomaly_cols": ["col_1", "col_2", ...],
  "anomaly_names": ["Forward Head", "Trunk Flexion", ...]
}
```

> ⚠️ **Note:** Model files (`.txt`, `.pkl`) are **not committed** to git due to size. Train your own models using the CSV data collected by the system, or contact the maintainer for the pre-trained weights.

---

## 📐 Ergonomic Scoring (RULA & REBA)

### RULA (Rapid Upper Limb Assessment)
Computed bilaterally (left/right) from:
- Shoulder flexion / abduction
- Elbow flexion
- Wrist flexion / deviation / pronation
- Neck flexion / lateral bend / rotation
- Trunk flexion / lateral bend / rotation

**Action Levels:**
| Score | Action |
|---|---|
| 1–2 | Acceptable |
| 3–4 | Further investigation required |
| 5–6 | Change soon |
| 7+ | Change immediately |

### REBA (Rapid Entire Body Assessment)
Extends RULA to include:
- Thigh / knee angles
- Load / coupling / activity adjustments

**Action Levels:**
| Score | Risk Level |
|---|---|
| 1 | Negligible |
| 2–3 | Low |
| 4–7 | Medium |
| 8–10 | High |
| 11+ | Very High — Immediate action |

---

## 📄 Report Generation

Click **"Generate Report"** on the `/reports` page. The system will:

1. Load the latest session CSV file
2. Run the AI model across all frames (sliding window)
3. Compute joint angle statistics and risk score histograms
4. Generate a multi-page PDF report including:
   - Cover page with session metadata and overall risk badge
   - Executive summary table
   - Joint angle statistics (min / max / mean / std / 95th percentile)
   - RULA / REBA score breakdown with action levels
   - AI predictive insights (LightGBM + Isolation Forest)
   - Trend charts (risk timeline, joint angle plots, RULA/REBA history)
   - Clinical recommendations

Reports are saved in `reports/` and available for download via the dashboard.

---

## 🔐 User Roles & Authentication

The system uses Flask session-based authentication with two hard-coded demo roles:

| Role | Email | Password | Access |
|---|---|---|---|
| **Doctor** | `doctor@exemple.com` | `doctor123` | Full access — AI, CSV, Reports, RULA, REBA |
| **Patient** | `patient@exemple.com` | `patient123` | Dashboard + System status only |

> ⚠️ **Production Note:** Replace the hard-coded credentials with a proper database-backed authentication system before deploying in a clinical environment. Change `app.secret_key` in `app.py`.

---

## 🧪 Testing

### Unit Tests

```bash
# Core logic tests
python test_logic.py

# Backend API tests (requires server running)
python test_backend.py

# Sensor combinatorial tests
python test_combinatorial_sensors.py
```

### Load / Stress Testing

```bash
python stress_test.py
```

The stress tester simulates concurrent ESP32 clients posting data at 10 Hz and reports throughput, latency, and error rates.

---

## 🌍 Deployment

### Local Network (LAN)

The server binds to `0.0.0.0:5000` by default — accessible to all devices on the same Wi-Fi network at the host machine's LAN IP.

### Production Recommendations

1. **Reverse Proxy**: Use Nginx or Apache in front of Flask
2. **WSGI Server**: Replace Flask dev server with Gunicorn + eventlet:
   ```bash
   pip install gunicorn eventlet
   gunicorn --worker-class eventlet -w 1 app:app
   ```
3. **HTTPS**: Enable SSL/TLS via Let's Encrypt (Certbot)
4. **Credentials**: Move all secrets to environment variables (`.env` + `python-dotenv`)
5. **Authentication**: Replace demo credentials with a proper user database

### ☁️ Cloud Deployment (Render / Heroku)

Due to the requirements of continuous WebSockets (Socket.IO) and AI model memory footprints, Serverless platforms (like Vercel or Netlify) are **not suitable**. 

We recommend deploying the application on **Render**:
1. Connect your GitHub repository on [Render.com](https://render.com/).
2. Create a new **Web Service**.
3. Build Command: `pip install -r requirements.txt`
4. Start Command: `python app.py`

See `READY_TO_DEPLOY.md` for full step-by-step instructions.

---

## 📊 Data Flow Summary

```
ESP32 IMU (10 Hz)
    → POST /api/data  OR  Firebase RTDB write
        → DataProcessor.process_incoming()
            → angle_math.compute_joint_angles()
            → RiskEngine.compute_risk()
            → RULAEngine.compute_side() × 2
            → REBAEngine.compute_side() × 2
            → FeatureExtractor.extract()
            → AIModels.predict()
            → CSVLogger.log()
            → socketio.emit('angles', payload)
                → Browser Dashboard updates in real time
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "Add my feature"`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Charrada** — [GitHub](https://github.com/charrada1993)

---

<div align="center">
Made with ❤️ for occupational health and ergonomic safety
</div>
