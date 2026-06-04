# Ergo Sensor v3.0 - Structure du projet

Ce document présente la structure globale du dépôt Ergo Sensor, expliquant le but de chaque composant majeur, répertoire et fichier critique du système.

## Vue d'ensemble de l'architecture globale
Le système Ergo Sensor est conçu comme un pipeline IoT de la périphérie vers le cloud. Il ingère des données cinématiques à haute fréquence provenant de capteurs matériels, les traite via des cadres ergonomiques cliniques (RULA/REBA) et un moteur d'IA d'apprentissage automatique, et visualise les mesures de risque dans un tableau de bord web en temps réel. Enfin, il génère des rapports PDF cliniques automatisés.

## Structure des répertoires

```text
c:\MSD_System\
├── .git/                      # Contrôle de version
├── csv_data/                  # Stockage local pour les sessions cinématiques enregistrées
├── dist/                      # Actifs frontend compilés / bundles de production
├── Ergo_Sensor_Project_Report/# Actifs de documentation académique/de projet
├── logs/                      # Journaux de fonctionnement du système
├── models/                    # Modèles d'IA sérialisés, métadonnées et tracés d'évaluation
├── PFE_LaTeX_Template/        # Code source LaTeX pour le rapport universitaire final
├── plots/                     # Images statiques servies au tableau de bord Web
├── reports/                   # Rapports PDF cliniques générés
├── static/                    # Actifs statiques frontend (CSS, JS, polices, images)
├── templates/                 # Modèles HTML frontend (Jinja2)
└── (Scripts Python racine)    # Logique backend principale
```

## Modules et fichiers de base

### 1. Application Web et routage
Ces fichiers gèrent le serveur web, le routage HTTP et la communication WebSocket en temps réel.
*   **`app.py`** : Le point d'entrée principal de l'application web Flask. Initialise le serveur, définit les routes de l'API (`/api/ai-metrics`), sert les pages HTML et gère le cycle de vie de Socket.IO.
*   **`socket_manager.py`** : Gère les connexions WebSocket entrantes et diffuse les données aux clients web connectés.

### 2. Frontend (UI/UX)
Situé dans `templates/` et `static/`.
*   **`templates/index.html`** : Le tableau de bord principal en temps réel affichant le squelette 3D et les scores RULA/REBA en direct.
*   **`templates/ai.html`** : Le tableau de bord d'analyse prédictive de l'IA affichant les prévisions de risque, la détection d'anomalies et les tracés d'évaluation complets du modèle.
*   **`templates/reports.html`** : Interface pour visualiser et télécharger les rapports PDF générés.
*   **`templates/csv_view.html`** : Interface pour parcourir l'historique des données de session CSV.
*   **`static/style.css`** : La feuille de style principale définissant l'esthétique clinique haut de gamme en mode sombre, les animations et les mises en page réactives.

### 3. Traitement des données et moteurs ergonomiques
Ces scripts traitent les données entrantes brutes et appliquent les formules cliniques.
*   **`data_processor.py`** : Le coordinateur central qui reçoit les données brutes des capteurs, déclenche les calculs d'angle et orchestre la notation RULA/REBA.
*   **`angle_math.py`** : Contient les mathématiques vectorielles complexes pour convertir les quaternions/angles d'Euler bruts des capteurs en angles articulaires biomécaniques standard (flexion, extension, etc.).
*   **`rula_engine.py` / `reba_engine.py`** : Implémentations programmatiques des cadres de notation des risques cliniques Rapid Upper Limb Assessment et Rapid Entire Body Assessment.
*   **`rula_ref.py` / `reba_ref.py`** : Tables de référence et constantes utilisées par les moteurs ergonomiques.

### 4. Moteur d'IA v3.0-Production
Le cœur prédictif du système.
*   **`ai_engine.py`** : Le moteur d'inférence d'exécution. Charge les modèles `.txt` et `.pkl` entraînés et exécute les prédictions en direct (score de risque, condition, gravité, anomalies) sur les flux de données entrants.
*   **`feature_extractor.py`** : Prépare les données brutes pour l'IA. Génère le vecteur de 75 caractéristiques en calculant les moyennes/écarts-types mobiles sur 15 trames, les caractéristiques de retard et les dérivées de vitesse/accélération pour fournir un contexte temporel aux modèles.
*   **`retrain_v3.py`** : Le pipeline d'entraînement hautement optimisé. Gère le chargement des données, le réglage des hyperparamètres Optuna, la validation croisée TimeSeriesSplit, l'entraînement du modèle LightGBM, l'analyse de l'importance des caractéristiques SHAP et le tracé automatique.
*   **`retrain_scratch.py` / `retrain_improved.py`** : Scripts d'entraînement hérités/expérimentaux.
*   **`generate_eval_plots.py`** : Script autonome pour générer la suite d'évaluation à 10 tracés (ROC, PR, matrices de confusion) sans réentraîner les modèles.

### 5. Rapports et journalisation
*   **`report_generator.py`** : Utilise ReportLab pour générer des documents PDF A4 de qualité clinique très détaillés. Il intègre des graphiques, des résumés RULA/REBA et des informations d'IA.
*   **`csv_logger.py`** : Gère l'écriture sécurisée des flux de capteurs à haute fréquence en direct dans des fichiers CSV locaux dans le répertoire `csv_data/` pour un stockage permanent et un réentraînement ultérieur.

### 6. Intégration IoT et base de données
*   **`firebase_listener.py`** : Se connecte à la base de données Firebase Realtime Database. Écoute les nouvelles données cinématiques poussées par le matériel ESP32 et les achemine vers le `data_processor.py` Python.

### 7. Configuration et utilitaires
*   **`config.py`** : Fichier de configuration centralisé stockant les clés de base de données, les chemins des modèles et les paramètres du système.
*   **`requirements.txt`** : Définit les dépendances du package Python requises pour exécuter le backend.
*   **`msd-monitor-system-firebase...json`** : La clé privée du compte de service utilisée pour s'authentifier auprès de Firebase.
*   **`condition_mappings.json`** : Mappe les étiquettes entières utilisées par le classificateur d'IA vers des conditions médicales lisibles par l'homme (ex: `0 -> normal`, `1 -> canal_carpien`).

## Le flux de données
1.  **Matériel** : Les capteurs IMU envoient des données à un ESP32, qui les pousse vers Firebase RTDB.
2.  **Ingestion** : `firebase_listener.py` détecte les nouvelles données et les envoie à `data_processor.py`.
3.  **Traitement** : `angle_math.py` calcule les articulations, puis `rula_engine.py` et `reba_engine.py` calculent le risque ergonomique immédiat.
4.  **Inférence d'IA** : Les données sont transmises à `feature_extractor.py` pour créer la fenêtre de 75 caractéristiques, qui est ensuite fournie à `ai_engine.py` pour prédire le risque à long terme et les anomalies.
5.  **Diffusion** : La charge utile complète (angles, RULA/REBA, prédictions d'IA) est diffusée via `socket_manager.py`.
6.  **Visualisation** : Le navigateur web reçoit la charge utile et met à jour le squelette 3D et les graphiques dans `index.html` et `ai.html`.
7.  **Rapports (Optionnel)** : Lorsqu'une session se termine, `report_generator.py` analyse le CSV enregistré et génère un rapport PDF.
