# Ergo Sensor v3.0 - Pile Technologique

Ce document présente les technologies, bibliothèques et frameworks de base utilisés pour construire le pipeline d'IA et l'application web Ergo Sensor, en précisant où chaque élément technologique est appliqué au sein du système.

## 1. Backend et Infrastructure Serveur
Le système central est construit sur un backend Python asynchrone haute performance conçu pour gérer les flux de capteurs en temps réel.

*   **Python (3.9+)** : Le langage de programmation principal pour l'ensemble du backend, du traitement des données et du pipeline d'IA.
*   **Flask** : Le framework web léger utilisé pour servir le tableau de bord frontend, les points de terminaison de l'API REST et gérer le routage (`app.py`).
*   **Flask-SocketIO / eventlet / gevent** : Utilisés pour la communication bidirectionnelle en temps réel entre le backend Python et le tableau de bord frontend. Cela permet la diffusion en direct des données du squelette 3D et des prédictions de l'IA sans rafraîchissement de la page.
*   **Gunicorn** : Le serveur HTTP WSGI de production utilisé pour déployer l'application Flask.
*   **SDK Admin Firebase** : Utilisé pour établir une connexion sécurisée en temps réel avec la base de données Firebase Realtime Database (`firebase_listener.py`) afin de recevoir les données cinématiques en direct des capteurs IoT.

## 2. Intelligence Artificielle et Apprentissage Automatique
Le "moteur d'IA Ergo Sensor v3.0-Production" est un ensemble multi-modèle conçu pour prédire les risques de troubles musculosquelettiques (TMS) et détecter les anomalies posturales.

*   **LightGBM** : Le framework de Boosting de Gradient de base utilisé pour les principaux modèles prédictifs en raison de sa vitesse et de sa grande précision sur les données cinématiques tabulaires.
    *   *Régresseur* : Prédit les scores de risque continus sur 10 jours.
    *   *Classificateur de condition* : Utilise le booster GBDT/DART pour classer 18 pathologies médicales distinctes (ex: canal carpien, hernie discale lombaire).
    *   *Classificateurs de gravité et par articulation* : Utilisés pour la détection granulaire d'anomalies par partie du corps.
*   **Scikit-Learn (sklearn)** : Fournit les utilitaires de base de l'apprentissage automatique.
    *   *TimeSeriesSplit* : Utilisé pour la validation croisée pendant l'entraînement afin de garantir l'absence de fuite de données temporelles.
    *   *Mesures* : Génère des matrices de confusion, des courbes ROC et des scores F1/Précision/RMSE.
*   **Optuna** : Un framework d'optimisation des hyperparamètres (HPO) utilisé dans `retrain_v3.py` pour rechercher automatiquement les paramètres les plus optimaux pour les modèles LightGBM.
*   **SHAP (SHapley Additive exPlanations)** : Utilisé via le `TreeExplainer` pour fournir une IA explicable. Il calcule l'importance des caractéristiques, révélant exactement *quelles* articulations et quels mouvements stimulent les prédictions de risque de l'IA.
*   **Pandas et NumPy** : La colonne vertébrale de la manipulation des données, utilisée de manière intensive dans `feature_extractor.py` et `retrain_v3.py` pour le calcul des fenêtres mobiles, des retards et des dérivées d'accélération à partir des données de séries temporelles brutes.

## 3. Traitement Ergonomique et Biomécanique
Avant que les données n'atteignent l'IA, elles sont traitées via des cadres ergonomiques cliniques validés.

*   **Mathématiques Python personnalisées (`angle_math.py`)** : Utilisent les mathématiques vectorielles et la trigonométrie pour calculer les angles articulaires 3D (flexion, extension, déviation) à partir des données quaternion/Euler IMU brutes.
*   **RULA (Rapid Upper Limb Assessment)** : Implémenté par programme (`rula_engine.py`) pour noter la tension posturale du haut du corps.
*   **REBA (Rapid Entire Body Assessment)** : Implémenté par programme (`reba_engine.py`) pour noter le risque postural du corps entier.

## 4. Frontend et Interface Utilisateur
Le tableau de bord est conçu pour être une application web moderne "sans dépendance", privilégiant les technologies web vanille aux frameworks lourds pour garantir des performances maximales et une latence minimale.

*   **HTML5 / Jinja2** : Les modèles sont servis par Flask et remplis de variables côté serveur avant le rendu.
*   **CSS3 Vanille** : Utilisé exclusivement pour le style. Comprend des variables CSS avancées, le glassmorphisme (`backdrop-filter`), CSS Grid/Flexbox et des micro-animations d'images clés pour créer une interface clinique de qualité supérieure en mode sombre (`static/style.css`, styles en ligne dans `ai.html`).
*   **JavaScript Vanille (ES6)** : Gère la logique côté client, l'ingestion de messages WebSocket et les mises à jour du DOM sans la surcharge de React ou Vue.
*   **Client Socket.IO** : Se connecte au serveur Flask-SocketIO pour recevoir des charges utiles JSON en direct des données des capteurs et des prédictions de l'IA.
*   **Three.js / WebGL (via dépendances)** : Utilisé pour effectuer le rendu du squelette en bâtonnets 3D en direct sur le tableau de bord principal (`index.html`).
*   **FontAwesome** : Fournit les icônes vectorielles évolutives utilisées dans toute l'interface utilisateur.
*   **Google Fonts** : Utilise 'Syne' et 'JetBrains Mono' pour une typographie moderne.

## 5. Génération de Rapports Médicaux
Le système génère des évaluations PDF automatisées de qualité clinique.

*   **ReportLab (Platypus)** : Une bibliothèque Python robuste utilisée dans `report_generator.py` pour construire par programme des documents PDF complexes de plusieurs pages avec des tableaux stylisés, des en-têtes, des pieds de page et un flux de texte.
*   **Matplotlib** : Utilisé sans état (backend `Agg`) pour générer des graphiques PNG de haute qualité (courbes d'apprentissage, tendances d'angle articulaire, courbes ROC) qui sont ensuite intégrés directement dans les rapports PDF et servis au tableau de bord d'IA (`generate_eval_plots.py`, `retrain_v3.py`).

## 6. Intégration Matérielle et Périphérique (Contexte supposé)
Bien que ce dépôt se concentre sur le logiciel, celui-ci est conçu pour s'interfacer avec un matériel périphérique spécifique.

*   **Microcontrôleurs ESP32** : Utilisés pour collecter les données des capteurs physiques.
*   **Capteurs IMU (BNO085 / MPU6050)** : Fournissent les données d'orientation 9-DOF brutes.
*   **NVIDIA Jetson Orin / Raspberry Pi** : Appareils périphériques où le pipeline de traitement Python peut être déployé pour un calcul sur site.

## 7. Cloud et Déploiement
*   **Render.com** : La cible PaaS (Platform as a Service) pour le déploiement de l'application web Flask.
*   **Firebase Realtime Database (RTDB)** : Agit comme le courtier de messages haute vitesse entre les capteurs IoT physiques et le backend Python déployé.
