# API Documentation

This document describes the internal and external communication protocols for the Ergo Sensor system.

---

## 📡 REST API Reference

The backend serves as a data ingestion and metrics hub.

### 1. Data Ingestion
`POST /api/data`
Used by ESP32 sensors or synthetic generators to push raw orientation data.

**Request Body (JSON):**
```json
{
  "sensor_id": "NECK",
  "roll": 10.5,
  "pitch": -5.2,
  "yaw": 0.0,
  "timestamp": 1685863200000
}
```

### 2. Sensor Status
`GET /api/sensors`
Returns the connectivity status and battery levels of all registered nodes.

### 3. Calibration
`POST /api/calibrate`
Triggers the zero-offset calibration for all currently active sensors.

### 4. AI Metrics
`GET /api/ai-metrics`
Returns the latest predicted risk scores and classified pathologies.

---

## ⚡ WebSocket (Socket.IO) Events

Real-time telemetry is handled via Socket.IO for sub-50ms latency.

### Outgoing Events (Server -> Client)

#### `angles`
Broadcasted at 10Hz. Contains processed joint angles and ergonomic scores.
- **Payload**:
    - `Neck_Flexion`: float
    - `Shoulder_R_Abduction`: float
    - `RULA_Score_R`: int
    - `REBA_Score_R`: int

#### `ai_prediction`
Broadcasted whenever an inference is complete.
- **Payload**:
    - `risk_score`: float [0.0 - 1.0]
    - `condition`: string (e.g., "Carpal Tunnel")
    - `severity`: string ("Low", "Medium", "High")

#### `raw_sensors`
Used for debugging raw IMU output.

---

## 📂 File Exports

### CSV Logs
`GET /api/csv/download/<session_id>`
Downloads the full biomechanical dataset for a specific session.

### PDF Reports
`GET /api/reports/download/<report_id>`
Downloads the generated clinical report in A4 format.

---

## 🔒 Security
In production, all endpoints are protected via **Firebase Authentication** tokens and **CORS** restrictions defined in `config.py`.
