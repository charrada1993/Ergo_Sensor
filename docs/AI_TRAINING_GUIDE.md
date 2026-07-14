# 🧠 Guide d'Entraînement du Modèle d'IA — Étape par Étape

Ce guide détaille le processus complet pour créer, optimiser et valider les modèles d'IA du système **Ergo Sensor** (v3.0-Production).

---

## 📋 Prérequis

Avant de commencer, assurez-vous d'avoir installé les dépendances de développement :
```bash
pip install -r requirements.txt
```

---

## 🚀 Étape 1 : Préparation des Données

Le modèle repose sur le fichier `dataset_TMS_enriched.csv`. Si vous souhaitez utiliser de nouvelles données :
1. Placez vos fichiers de télémétrie brute dans le dossier `csv_data/`.
2. Assurez-vous que les colonnes correspondent aux 12 angles articulaires cliniques.
3. Le script `feature_extractor.py` se chargera de transformer ces données brutes en un vecteur de **75 caractéristiques**.

---

## 🛠️ Étape 2 : Extraction des Caractéristiques (Feature Engineering)

Le système ne travaille pas sur des images, mais sur des séries temporelles biomécaniques. 
- **Action** : Le script de réentraînement appelle automatiquement `FeatureExtractor`.
- **Ce qui est calculé** : 
    - Moyennes et variances mobiles (fenêtre de 1.5s).
    - Valeurs de retard (Lag).
    - Vitesses et accélérations angulaires.
    - Asymétries gauche/droite.

---

## 🎯 Étape 3 : Optimisation des Hyperparamètres avec Optuna

Pour obtenir la meilleure précision sans surapprentissage (overfitting) :
- **Outil** : [Optuna](https://optuna.org/)
- **Processus** : Le script lance 10 essais (trials) par modèle pour trouver la combinaison idéale de `num_leaves`, `learning_rate`, `feature_fraction`, etc.
- **Validation** : Utilisation de `TimeSeriesSplit` pour garantir que le modèle apprend la chronologie des mouvements.

---

## 🏗️ Étape 4 : Entraînement de l'Ensemble de Modèles

Exécutez la commande suivante pour lancer tout le pipeline :
```bash
python retrain_v3.py
```

Ce script va entraîner 4 types de modèles :
1. **Régresseur** : Pour le score de risque continu [0-1].
2. **Classificateur de Condition** : Pour identifier les 18 pathologies.
3. **Classificateur de Gravité** : (Faible, Moyen, Élevé).
4. **Détecteurs d'Anomalies** : 5 modèles binaires pour chaque articulation critique.

---

## 📊 Étape 5 : Évaluation et Diagnostics

Une fois l'entraînement terminé, consultez le dossier `models/` et `plots/`. Le système génère automatiquement :
- **Matrices de Confusion** : Pour voir les erreurs de classification des pathologies.
- **Courbes ROC et PR** : Pour évaluer la sensibilité des alertes.
- **Importance des Caractéristiques (SHAP)** : Pour comprendre quels angles influencent le plus le risque.

---

## 🔄 Étape 6 : Déploiement du Modèle

Les modèles entraînés sont sauvegardés sous forme de fichiers `.txt` (LightGBM) et `.pkl` (Scalers) dans le dossier `models/`.
- **Activation** : Le moteur `ai_engine.py` chargera automatiquement ces nouveaux fichiers au prochain démarrage du serveur `app.py`.

---

## 💡 Conseils d'Expert
- **Déséquilibre** : Si une pathologie est mal détectée, augmentez son poids dans le dictionnaire `class_weight` du script `retrain_v3.py`.
- **Latence** : Ne dépassez pas 100-200 arbres pour garder une inférence temps réel fluide sur les petits serveurs.
