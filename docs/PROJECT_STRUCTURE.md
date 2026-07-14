# Ergo Sensor v3.0 - Structure du Projet

Ce document détaille l'organisation du dépôt Ergo Sensor, expliquant le rôle de chaque module et répertoire clé.

---

## 📂 Arborescence du Système

```text
c:\MSD_System\
├── docs/                      # Documentation centralisée (Guide, IA, API, Rapports)
├── static/                    # Actifs frontend (CSS, JS, Images, Polices)
│   ├── dashboard.js           # Logique temps réel du tableau de bord
│   ├── style.css              # Design Dark-Mode clinique
│   └── 3d_view.js             # Rendu Three.js du squelette
├── templates/                 # Modèles HTML (Jinja2)
│   ├── index.html             # Tableau de bord principal
│   ├── ai.html                # Interface d'analyse prédictive
│   └── calibration.html       # Poste de travail d'étalonnage
├── models/                    # Modèles d'IA sérialisés (.txt, .pkl) et métadonnées
├── csv_data/                  # Base de données locale (Séries temporelles CSV)
├── logs/                      # Journaux système et erreurs
├── reports/                   # Rapports PDF cliniques générés
├── app.py                     # Point d'entrée Flask et routage
├── data_processor.py          # Orchestrateur central des flux de données
├── ai_engine.py               # Moteur d'inférence LightGBM
├── angle_math.py              # Noyau de calcul cinématique
├── feature_extractor.py       # Générateur de vecteurs (75 features)
├── rula_engine.py             # Moteur de notation RULA
├── reba_engine.py             # Moteur de notation REBA
├── firebase_listener.py       # Écouteur de flux IoT (Firebase RTDB)
├── csv_logger.py              # Gestionnaire de persistance thread-safe
├── report_generator.py        # Générateur de documents PDF
└── retrain_v3.py              # Pipeline d'entraînement et optimisation Optuna
```

---

## 🛠️ Modules de Base

### 1. Ingestion et Flux Temps Réel
- **`app.py`** : Gère le serveur HTTP et les espaces de noms Socket.IO.
- **`firebase_listener.py`** : S'abonne aux événements Firebase pour une latence minimale.
- **`socket_manager.py`** : Diffuse les mesures traitées aux clients web.

### 2. Moteur de Calcul (Le "Cœur")
- **`angle_math.py`** : Transforme les quaternions bruts en angles articulaires cliniques.
- **`rula_engine.py` / `reba_engine.py`** : Applique les algorithmes de notation ergonomique validés.
- **`data_processor.py`** : Synchronise les capteurs et déclenche les inférences.

### 3. Intelligence Artificielle (L' "Intelligence")
- **`ai_engine.py`** : Exécute l'ensemble de modèles en temps réel.
- **`feature_extractor.py`** : Construit la fenêtre temporelle dynamique de 1.5s (15 trames).
- **`retrain_v3.py`** : Gère le réentraînement et la validation `TimeSeriesSplit`.

### 4. Sortie et Reporting (La "Valeur")
- **`report_generator.py`** : Produit des rapports de qualité médicale intégrant les insights de l'IA.
- **`csv_logger.py`** : Assure que chaque mouvement est enregistré pour l'audit futur.

---

## 🔄 Flux de Travail Technique

1. **Capture** : L'ESP32 envoie les données à Firebase.
2. **Traitement** : Le backend Python convertit, note et prédit.
3. **Visualisation** : Socket.IO pousse les données vers le tableau de bord 3D.
4. **Audit** : Un PDF est généré à la fin de la session pour le clinicien.
