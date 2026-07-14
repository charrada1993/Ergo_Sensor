# 🦾 Ergo Sensor : Un Cadre Complet Piloté par l'IA pour l'Évaluation des Risques de Troubles Musculosquelettiques

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-black?style=for-the-badge&logo=flask&logoColor=white)
![Firebase](https://img.shields.io/badge/Firebase-Realtime-orange?style=for-the-badge&logo=firebase&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-AI-green?style=for-the-badge)
![Render](https://img.shields.io/badge/Render-Cloud-00d4ff?style=for-the-badge&logo=render&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)

**Une plateforme de surveillance biomécanique de bout en bout, à haute fréquence, conçue pour la sécurité et la santé au travail en temps réel, utilisant des réseaux de capteurs distribués et l'apprentissage automatique d'ensemble.**

</div>

---

## 📑 Table des matières

1. [Résumé analytique](#1-résumé-analytique)
2. [Contexte clinique et méthodologie](#2-contexte-clinique-et-méthodologie)
    - 2.1 RULA (Rapid Upper Limb Assessment)
    - 2.2 REBA (Rapid Entire Body Assessment)
3. [Architecture du système](#3-architecture-du-système)
4. [Infrastructure matérielle](#4-infrastructure-matérielle)
    - 4.1 Configuration du microcontrôleur ESP32
    - 4.2 Stratégie de placement des capteurs IMU
5. [Pile logicielle et technologies](#5-pile-logicielle-et-technologies)
6. [Modélisation biomécanique (vecteur de 75 caractéristiques)](#6-modélisation-biomécanique-vecteur-de-75-caractéristiques)
7. [Ensemble d'Intelligence Artificielle (v3.0-Production)](#7-ensemble-dintelligence-artificielle-v30-production)
8. [Poste de travail d'étalonnage en direct et pupitre de télémétrie](#8-poste-de-travail-détalonnage-en-direct-et-pupitre-de-télémétrie)
9. [Guide de questions-réponses pour la soutenance de projet](#9-guide-de-questions-réponses-pour-la-soutenance-de-projet)
10. [Documentation technique module par module](#10-documentation-technique-module-par-module)
11. [Pipeline de données et cycle de vie](#11-pipeline-de-données-et-cycle-de-vie)
12. [Rapports et perspectives cliniques](#12-rapports-et-perspectives-cliniques)
13. [Installation et configuration locale](#13-installation-et-configuration-locale)
14. [Déploiement sur le cloud (Render.com)](#14-déploiement-sur-le-cloud-rendercom)
15. [Dépannage et débogage](#15-dépannage-et-débogage)
16. [Feuille de route future et recherche](#16-feuille-de-route-future-et-recherche)
17. [Glossaire des termes](#17-glossaire-des-termes)
18. [Licence et contribution](#18-licence-et-contribution)

---

## 1. Résumé analytique

**Ergo Sensor** représente la prochaine génération de surveillance de la santé au travail. Les audits ergonomiques traditionnels sont intermittents, subjectifs et réactifs. Ils surviennent souvent après qu'un travailleur a déjà développé des symptômes d'un trouble musculosquelettique (TMS).

Ergo Sensor inverse ce paradigme en fournissant une **surveillance continue, objective et prédictive**. En déployant un réseau d'unités de mesure inertielle (IMU) sur tout le corps, le système capture la cinématique du corps entier à 10 Hz. Ces données brutes sont transformées en angles articulaires cliniques, notées par rapport aux normes ergonomiques internationales, et traitées par un moteur d'IA sophistiqué pour prédire les risques de blessures futurs.

Propositions de valeur clés :
- **Rétroaction à latence nulle** : Alertes immédiates pour les postures dangereuses.
- **Audit objectif** : Supprime les biais humains des scores RULA/REBA.
- **Analytique prédictive** : Prévoit le risque de TMS sur un horizon de travail de 10 jours.
- **Jumeau numérique 3D** : Visualisation en direct du squelette 3D de la cinématique du travailleur.
- **Évolutivité** : Déployable via une infrastructure cloud pour surveiller des usines entières.

---

## 2. Contexte clinique et méthodologie

Le système repose sur deux piliers de la science ergonomique :

### 2.1 RULA (Rapid Upper Limb Assessment)
RULA est une méthode d'enquête développée pour être utilisée dans les enquêtes ergonomiques sur les lieux de travail où les troubles des membres supérieurs sont fréquents. Ergo Sensor automatise cela en :
- Surveillant les postures du cou, du tronc et des membres supérieurs.
- Prenant en compte les mouvements répétitifs et la charge statique.
- Générant un score de 1 (acceptable) à 7 (changement immédiat requis).

### 2.2 REBA (Rapid Entire Body Assessment)
REBA est spécifiquement conçu pour évaluer les tâches où les postures sont dynamiques, imprévisibles ou instables.
- Comprend l'évaluation des membres inférieurs (jambes et pieds).
- Prend en compte le couplage et le poids de la charge.
- Fournit un profil de risque complet de Négligeable à Très élevé.

Ergo Sensor implémente ceux-ci comme des **Moteurs bilatéraux**, calculant les scores pour les côtés gauche et droit du corps simultanément dans [rula_engine.py](file:///c:/MSD_System/rula_engine.py) et [reba_engine.py](file:///c:/MSD_System/reba_engine.py) pour détecter les déséquilibres posturaux.

---

## 3. Architecture du système

L'architecture suit un modèle distribué et piloté par les événements :

1.  **Couche de capteurs** : Les appareils ESP32 collectent des données quaternion/Euler à partir des IMU.
2.  **Ingestion cloud** : Les données sont transmises via HTTP/JSON ou Firebase Realtime Database.
3.  **Moteur de traitement** : Un backend basé sur Python effectue :
    - La conversion **Quaternions vers Angles** en utilisant une logique trigonométrique géométrique dans [angle_math.py](file:///c:/MSD_System/angle_math.py).
    - L'**Ingénierie des caractéristiques** pour construire le vecteur biomécanique de 75 caractéristiques dans [feature_extractor.py](file:///c:/MSD_System/feature_extractor.py).
    - L'**Inférence d'IA** en utilisant des boosters LightGBM pré-entraînés gérés dans [ai_engine.py](file:///c:/MSD_System/ai_engine.py).
4.  **Couche de distribution** : Les résultats sont émis via Socket.IO vers les clients web connectés.
5.  **Couche de persistance** : Chaque trame est enregistrée dans un fichier CSV enrichi pour l'entraînement futur et les journaux d'audit.

---

## 4. Infrastructure matérielle

### 4.1 Configuration du microcontrôleur ESP32
Le système utilise l'**ESP32-WROOM-32** pour son traitement double cœur et son Wi-Fi intégré. 
- **Taux d'échantillonnage** : 10 Hz (intervalles de 100 ms).
- **Communication** : JSON via HTTP ou SDK Firebase.
- **Alimentation** : Batterie LiPo (500 mAh recommandée pour des quarts de travail de 8 heures).

### 4.2 Stratégie de placement des capteurs IMU
Pour une évaluation complète à 12 capteurs, les capteurs doivent être placés :
- **Axial** : Tête/Cou (C7), Haut du dos/Tronc (T12).
- **Membres supérieurs** : Biceps, avant-bras, mains bilatéraux.
- **Membres inférieurs** : Cuisses, jambes bilatérales.

---

## 5. Pile logicielle et technologies

- **Backend** : Python 3.11 (Stabilité et écosystème ML).
- **Cadre Web** : Flask 3.0 (Léger et extensible).
- **Moteur en temps réel** : Flask-SocketIO + Gevent (WebSockets à haute concurrence).
- **Base de données** : Firebase RTDB (Synchronisation cloud à faible latence).
- **Apprentissage automatique** : 
    - **LightGBM** : Boosting de gradient rapide pour les données tabulaires.
    - **Scikit-learn** : Prétraitement et validation TimeSeriesSplit.
    - **SHAP** : Interprétabilité du modèle via TreeExplainer.
- **Visualisation** : 
    - **Three.js / WebGL** : Gréement et rendu 3D direct du squelette du travailleur.
    - **Chart.js** : Widgets et graphiques de tableau de bord en temps réel.
- **Rapports** : ReportLab (Génération de PDF A4 haute fidélité).

---

## 6. Modélisation biomécanique (vecteur de 75 caractéristiques)

Le "cerveau" central du système est le **vecteur de caractéristiques biomécaniques**. Toutes les 100 ms, le système génère une description de série temporelle à **75 dimensions** de l'état du corps (v3.0-Production) :

1.  **Angles cinématiques (12)** : Rotations articulaires brutes (flexion du cou, abduction de l'épaule, etc.).
2.  **Moyennes mobiles (12)** : Moyenne temporelle sur 15 trames des articulations centrales, capturant les postures soutenues.
3.  **Variances mobiles (12)** : Écart-type sur 15 trames, capturant la gigue de mouvement et les micro-vibrations.
4.  **Caractéristiques de retard (12)** : L'angle de l'articulation il y a exactement 15 trames (1,5 seconde).
5.  **Asymétrie bilatérale (5)** : Différence absolue entre les articulations droite et gauche (épaule, coude, poignet, hanche, genou).
6.  **Dynamique de la vitesse (7)** : Taux de changement pour les articulations majeures (degrés par seconde).
7.  **Proxys d'énergie (7)** : Scores d'interaction vitesse × durée par articulation.
8.  **Charge composite (2)** : Scores de charge pondérés pour le haut et le bas du corps.
9.  **Drapeaux de risque élevé (3)** : Indicateurs binaires pour l'hyperflexion/hyperextension.
10. **Superpositions de degrés bruts (2)** : Angles bruts non normalisés pour des articulations critiques spécifiques.
11. **Accélérations (5)** : Dérivée de la vitesse agrégée (changement de degrés par seconde carrée).

Ce vecteur permet à l'IA de comprendre non seulement *où* se trouvent les articulations, mais aussi *à quelle vitesse* elles bougent, leur historique temporel au cours des 1,5 dernières secondes, et *à quel point* la posture actuelle est inhabituelle.

---

## 7. Ensemble d'Intelligence Artificielle (v3.0-Production)

Le pipeline d'IA v3.0-Production a été optimisé à l'aide d'**Optuna** (10 essais) et d'une validation croisée **TimeSeriesSplit** pour garantir une absence de fuite de données temporelles.

### 7.1 Prévision des risques à 10 jours LightGBM
Le modèle **Régresseur** (R²=0,9981) analyse les modèles de mouvement pour prédire le stress cumulé. Si un travailleur présente des angles à haut risque soutenus, la probabilité de risque à 10 jours augmente, alertant le clinicien pour prévenir l'épuisement potentiel ou les blessures chroniques.

### 7.2 🚶‍♂️ Visualisation du jumeau numérique 3D
Ergo Sensor v3.0 comprend un **squelette humanoïde 3D** en temps réel rendu directement dans le navigateur à l'aide de **Three.js**. 
- Mappe les angles de roulis/tangage/lacet sur un avatar 3D gréé.
- Permet aux cliniciens d'observer la posture du travailleur sous n'importe quel angle (rotation à 360°).
- Fournit une confirmation visuelle instantanée des anomalies détectées par l'IA.

### 7.3 Classificateurs posturaux granulaires
Cinq classificateurs LightGBM binaires dédiés (F1 moyen=0,9906) fournissent des courbes de probabilité en temps réel pour :
- **Hyperflexion du cou**
- **Hyperextension de l'épaule**
- **Tension du poignet**
- **Torsion du tronc**
- **Hyperextension du coude**

De plus, le **Classificateur de condition** (Précision=99,60 %) identifie 18 conditions pathologiques distinctes, et le **Classificateur de gravité** étiquette le niveau de risque.

### 7.4 Expliquabilité via SHAP
À l'aide de **SHAP TreeExplainer**, le système fournit une "interprétabilité locale". Pour chaque alerte à haut risque, le système identifie la "contribution" de chacune des 75 caractéristiques. 
- *Aperçu du clinicien* : "Le risque est élevé principalement en raison de la rotation extrême du tronc, et non de l'angle de l'épaule."

---

## 8. Poste de travail d'étalonnage en direct et pupitre de télémétrie

Ergo Sensor dispose d'un **poste de travail d'étalonnage** de qualité médicale de premier ordre à `/calibrate` (`templates/calibration.html`) :
*   **Sélecteur de rôle segmenté** : Onglets en ligne interactifs pour échanger les instructions entre le `Protocole clinicien` et la `Configuration du patient`.
*   **Scanner de posture animé** : Guide de posture encadré à l'intérieur d'un viseur de diagnostic scientifique dynamique, avec des lignes de mesure au néon et des animations de balayage laser.
*   **Superposition de compte à rebours interactive** : Modale plein écran directe (`RESTEZ IMMOBILE` -> `ACQUISITION DE LA LIGNE DE BASE` -> `ÉTALONNAGE VERROUILLÉ`) fournissant une confirmation visuelle immédiate de la mise à zéro de la ligne de base du capteur.
*   **Colonnes de diagnostic orthopédique** : Grille propre regroupant les angles en trois colonnes distinctes : *Colonne vertébrale*, *Membre supérieur droit* et *Membre supérieur gauche*.
*   **Télémétrie du coude à trois angles** : Mesures en temps réel pour la **Flexion, la Déviation latérale (Roulis)** et la **Rotation axiale (Lacet)** sur les deux avant-bras.
*   **Indicateurs de plage de mouvement dynamique (ROM)** : Barres de progression colorées dynamiquement selon la gravité posturale (Teal pour sûr, Orange pour prudence, Rouge pour hyperflexion à haut risque).
*   **Variables à double thème** : Feuille de style pilotée par variables complètement intégrée prenant en charge la commutation transparente entre les thèmes sombre et clair.

---

## 9. Guide de questions-réponses pour la soutenance de projet

Pour aider aux présentations universitaires ou professionnelles, un guide détaillé de revue académique et de préparation à la soutenance a été créé :
📄 **[FINAL_PROJECT_QA.md](file:///c:/MSD_System/FINAL_PROJECT_QA.md)**

Ce document contient 30 questions de niveau expert et des réponses détaillées couvrant la science médicale, la géométrie des capteurs, la concurrence en temps réel et les méthodologies d'apprentissage automatique de ce projet.

---

## 10. Documentation technique module par module

- **`app.py`** : Le système nerveux central. Gère le routage HTTP, l'authentification des utilisateurs et les espaces de noms Socket.IO.
- **`config.py`** : Référentiel central pour toutes les constantes, les ID de capteurs et les identifiants cloud.
- **`data_processor.py`** : L'orchestrateur principal. Reçoit les données, déclenche la conversion mathématique, exécute les modèles d'IA et enregistre dans CSV.
- **`ai_engine.py`** : Charge et gère le cycle de vie des modèles LightGBM et des explicateurs SHAP.
- **`firebase_listener.py`** : Un thread d'arrière-plan qui s'abonne aux événements Firebase pour un flux de données fluide du capteur vers le serveur.
- **`angle_math.py`** : Le noyau géométrique. Implémente les transformations d'angle d'Euler à partir des données IMU brutes.
- **`feature_extractor.py`** : Calcule le vecteur de 75 caractéristiques à l'aide de files d'attente à fenêtre glissante.
- **`report_generator.py`** : Génère des rapports PDF multipages contenant des statistiques, des informations d'IA et des courbes d'anomalies.
- **`csv_logger.py`** : Gère l'écriture thread-safe des ensembles de données biomécaniques sur le disque.

---

## 11. Pipeline de données et cycle de vie

1.  **Capture** : L'ESP32 lit les données MPU6050/9250.
2.  **Transport** : JSON via WebSocket sécurisé ou flux Firebase.
3.  **Ingestion** : `FirebaseListener` ou `/api/data` capture le paquet.
4.  **Normalisation** : Les données sont filtrées via `EXPECTED_SENSORS` dans `Config`.
5.  **Calcul** : Les angles, RULA/REBA et les probabilités d'IA sont calculés.
6.  **Diffusion** : `socketio.emit('angles', payload)` envoie les données au navigateur.
7.  **Persistance** : L'état entier est ajouté à un fichier CSV de session quotidienne.

---

## 12. Rapports et perspectives cliniques

Ergo Sensor génère des rapports de **"qualité clinique"**. Contrairement aux graphiques simples, ces rapports comprennent :
- **Courbes de probabilité d'anomalies** : Tracés de séries temporelles montrant exactement quand les seuils de risque ont été franchis.
- **Cartes thermiques articulaires** : Identification des régions corporelles les plus sollicitées.
- **Répartition statistique** : Valeurs P95, Moyenne et Max pour chaque articulation.
- **Suggestions cliniques** : Recommandations automatisées basées sur les niveaux d'action (ex: "Mettre en œuvre des étirements de 5 minutes toutes les 30 minutes").

---

## 13. Installation et configuration locale

### 13.1 Prérequis
- Python 3.11+
- Virtualenv
- Git

### 13.2 Étapes
```bash
git clone https://github.com/charrada1993/Ergo_Sensor.git
cd Ergo_Sensor
python -m venv .venv
source .venv/bin/activate  # Ou .venv\Scripts\activate sur Windows
pip install -r requirements.txt
python app.py
```

---

## 14. Déploiement sur le cloud (Render.com)

Render est la plateforme recommandée pour le déploiement en production.

### 14.1 Variables d'environnement
- `PYTHON_VERSION` : `3.11.0`
- `FIREBASE_CREDS_JSON` : Le contenu brut de votre clé JSON Firebase.
- `PORT` : (Géré par Render)

### 14.2 Commande de démarrage
```bash
gunicorn -k geventwebsocket.gunicorn.workers.GeventWebSocketWorker -w 1 app:app
```

---

## 15. Dépannage et débogage

- **Échec de la connexion WebSocket** : Assurez-vous d'utiliser le worker `gevent` et que votre pare-feu autorise le trafic WebSocket.
- **Erreur d'authentification Firebase** : Vérifiez que `FIREBASE_CREDS_JSON` est une chaîne JSON valide commençant par `{`.
- **Erreur de modèle manquant** : Assurez-vous que le répertoire `models/` contient tous les fichiers `.txt` et `.pkl`. Vérifiez `.gitignore`.
- **Problèmes de latence** : Réduisez le taux d'échantillonnage sur l'ESP32 (augmentez `POST_INTERVAL_MS`).

---

## 16. Feuille de route future et recherche

- **IA à la périphérie** : Déplacer l'inférence LightGBM directement sur l'ESP32 (modèle S3).
- **Compagnon mobile** : Application basée sur Flutter pour que les travailleurs voient leurs propres scores en direct.
- **Hybride vision par ordinateur** : Combinaison des données IMU avec la détection de profondeur par caméra OAK-D pour une précision de 100 %.
- **Intégration HRV** : Ajout de la variabilité de la fréquence cardiaque pour évaluer la tension physique interne.

---

## 17. Glossaire des termes

- **IMU** : Unité de mesure inertielle (accéléromètre + gyroscope).
- **RULA/REBA** : Normes internationales d'évaluation posturale.
- **TMS** : Trouble musculosquelettique.
- **SHAP** : Une méthode mathématique pour expliquer la sortie de n'importe quel modèle d'apprentissage automatique.
- **Gevent** : Une bibliothèque réseau Python basée sur les coroutines qui permet une haute concurrence pour les WebSockets.

---

## 18. Licence et contribution

Ce projet est sous licence **MIT**. Nous encourageons les forks et les contributions axés sur la précision biomécanique ou l'expérience utilisateur.

**Maintenu par Charrada** | [Profil GitHub](https://github.com/charrada1993)

---

<div align="center">
Conçu pour l'avenir du travail. Construit pour la sécurité des travailleurs.
</div>

*(Version du document 3.0-Production - IA Analytique prédictive et édition de séries temporelles)*
