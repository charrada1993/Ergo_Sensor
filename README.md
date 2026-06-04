# 🦾 Ergo Sensor : Évaluation des Risques Biomécaniques par l'IA

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-black?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Firebase](https://img.shields.io/badge/Firebase-Realtime-orange?style=flat-square&logo=firebase&logoColor=white)](https://firebase.google.com/)
[![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)](LICENSE)
[![Maintenance](https://img.shields.io/badge/Maintenu%3F-oui-green.svg?style=flat-square)](https://github.com/charrada1993/Ergo_Sensor/graphs/commit-activity)

**Ergo Sensor** est une plateforme de santé au travail de pointe, fonctionnant en temps réel. Elle exploite un réseau distribué de capteurs IoT et l'apprentissage automatique d'ensemble pour prédire et prévenir les Troubles Musculosquelettiques (TMS) dans les environnements industriels.

---

## 🚀 Fonctionnalités Clés

- **Surveillance Continue** : Capture de la cinématique du corps entier à 10 Hz via ESP32 et IMU.
- **Scoring Clinique** : Moteurs d'évaluation RULA et REBA automatisés et objectifs.
- **IA Prédictive** : Prévision du risque de blessure à 10 jours via des boosters LightGBM.
- **Jumeau Numérique 3D** : Visualisation en direct du squelette humanoïde directement dans le navigateur.
- **Analyses Actionnables** : Rapports PDF automatisés avec cartes thermiques articulaires et suggestions cliniques.
- **IA Explicable (XAI)** : Transparence pilotée par SHAP pour chaque alerte à haut risque.

---

## 📂 Documentation

La documentation du projet est organisée en modules spécialisés :

| Document | Description |
| :--- | :--- |
| 📖 [**Guide Principal**](./docs/README.md) | Vue d'ensemble complète de la méthodologie clinique et de l'architecture. |
| 🧠 [**Guide d'Entraînement IA**](./docs/AI_TRAINING_GUIDE.md) | **[NOUVEAU]** Guide étape par étape pour créer et optimiser les modèles. |
| 🤖 [**Moteur d'IA**](./docs/AI.md) | Plongée profonde dans les modèles ML, l'ingénierie des caractéristiques et SHAP. |
| 📊 [**Rapport de Performance**](./docs/AI_MODEL_REPORT.md) | Métriques détaillées, scores de validation et tracés de diagnostic v3.0. |
| 🛠️ [**Pile Technique**](./docs/TECH_STACK.md) | Détail des technologies Backend, Frontend et IoT utilisées. |
| 📐 [**Architecture**](./docs/PROJECT_STRUCTURE.md) | Structure des répertoires et documentation module par module. |
| ❓ [**Questions-Réponses**](./docs/FINAL_PROJECT_QA.md) | Guide de niveau expert pour la soutenance et les questions techniques. |
| 🏗️ [**Déploiement**](./docs/READY_TO_DEPLOY.md) | Guide étape par étape pour l'hébergement cloud sur Render.com. |
| 📡 [**Documentation API**](./docs/API_DOCUMENTATION.md) | Référence des points de terminaison REST et des événements Socket.IO. |
| 🔌 [**Configuration Matérielle**](./docs/HARDWARE_SETUP.md) | Guide pour assembler et configurer les nœuds capteurs. |

---

## 🛠️ Démarrage Rapide

### Prérequis
- Python 3.11+
- Virtualenv
- Git

### Installation
```bash
# Cloner le dépôt
git clone https://github.com/charrada1993/Ergo_Sensor.git
cd Ergo_Sensor

# Configurer l'environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt

# Lancer le serveur
python app.py
```
Accédez au tableau de bord à l'adresse `http://localhost:5000`.

---

## 🤝 Contribution et Communauté

Nous accueillons les contributions des cliniciens, des data scientists et des ingénieurs.
- 📄 [**Directives de Contribution**](./docs/CONTRIBUTING.md)
- 📜 [**Journal des Modifications**](./docs/CHANGELOG.md)

---

## 📜 Licence

Ce projet est sous licence **MIT**. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

**Maintenu par [Charrada](https://github.com/charrada1993)** | *Conçu pour la sécurité des travailleurs.*
