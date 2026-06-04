# 🦾 Ergo Sensor: AI-Driven Biomechanical Risk Assessment

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-black?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Firebase](https://img.shields.io/badge/Firebase-Realtime-orange?style=flat-square&logo=firebase&logoColor=white)](https://firebase.google.com/)
[![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)](LICENSE)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg?style=flat-square)](https://github.com/charrada1993/Ergo_Sensor/graphs/commit-activity)

**Ergo Sensor** is a cutting-edge, real-time occupational health platform. It leverages a distributed network of IoT sensors and ensemble machine learning to predict and prevent Musculoskeletal Disorders (MSDs) in industrial environments.

---

## 🚀 Key Features

- **Continuous Monitoring**: 10Hz full-body kinematics capture using ESP32 & IMUs.
- **Clinical Scoring**: Automated, objective RULA and REBA assessment engines.
- **Predictive AI**: 10-day injury risk forecasting using LightGBM boosters.
- **3D Digital Twin**: Live humanoid skeleton visualization directly in the browser.
- **Actionable Insights**: Automated PDF reports with joint heatmaps and clinical suggestions.
- **Explainable AI (XAI)**: SHAP-driven transparency for every high-risk alert.

---

## 📂 Documentation

The project documentation is organized into specialized modules:

| Document | Description |
| :--- | :--- |
| 📖 [**Main Guide**](./docs/README.md) | Comprehensive overview of the clinical methodology and system architecture. |
| 🧠 [**AI Engine**](./docs/AI.md) | Deep dive into machine learning models, feature engineering, and SHAP. |
| 📊 [**Performance Report**](./docs/AI_MODEL_REPORT.md) | Detailed metrics, validation scores, and diagnostic plots for v3.0. |
| 🛠️ [**Tech Stack**](./docs/TECH_STACK.md) | Breakdown of backend, frontend, and IoT technologies used. |
| 📐 [**Architecture**](./docs/PROJECT_STRUCTURE.md) | Directory structure and module-by-module documentation. |
| ❓ [**Project Q&A**](./docs/FINAL_PROJECT_QA.md) | Expert-level guide for project defense and technical inquiries. |
| 🏗️ [**Deployment**](./docs/READY_TO_DEPLOY.md) | Step-by-step guide for cloud hosting on Render.com. |

---

## 🛠️ Quick Start

### Prerequisites
- Python 3.11+
- Virtualenv
- Git

### Installation
```bash
# Clone the repository
git clone https://github.com/charrada1993/Ergo_Sensor.git
cd Ergo_Sensor

# Set up virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch the server
python app.py
```
Access the dashboard at `http://localhost:5000`.

---

## 🤝 Contributing & Community

We welcome contributions from clinicians, data scientists, and engineers.
- 📄 [**Contributing Guidelines**](./docs/CONTRIBUTING.md)
- 📜 [**Changelog**](./docs/CHANGELOG.md)
- 🐛 [**Bug Reports**](https://github.com/charrada1993/Ergo_Sensor/issues)

---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

**Maintained by [Charrada](https://github.com/charrada1993)** | *Designed for the safety of workers.*
