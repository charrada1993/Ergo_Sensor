# 🤖 Ergo Sensor — Package des modèles du moteur d'IA (v3.0-Production)

Ce répertoire contient les modèles d'apprentissage automatique sérialisés, les prétraitements et les tracés de diagnostic pour le **moteur d'IA Ergo Sensor v3.0-Production**.

---

## 🗂️ Manifeste des modèles

### 1. Modèles d'exposition et prédictifs de base
*   **`lgb_regressor.txt`** : Régresseur LightGBM prédisant la probabilité de blessure musculosquelettique continue sur 10 jours `[0,0 – 1,0]`. Optimisé avec Optuna et entraîné avec la validation croisée `TimeSeriesSplit`.
*   **`lgb_classifier.txt`** : Classificateur de condition à 18 classes LightGBM identifiant des conditions pathologiques spécifiques (ex: canal carpien, hernie discale lombaire, cervicalgie). Configuré avec des poids de classe équilibrés pour lutter contre l'asymétrie des données.
*   **`lgb_severity.txt`** : Classificateur de gravité à 3 classes LightGBM étiquetant la gravité du risque postural comme `faible`, `moyenne` ou `élevée`.
*   **`isolation_forest.pkl`** : Modèle de détection d'anomalies non supervisé entraîné pour identifier des configurations articulaires nouvelles ou pathologiquement irrégulières.

### 2. Classificateurs d'anomalies articulaires binaires
Cinq classificateurs LightGBM binaires dédiés entraînés pour déclencher des alertes lorsque des articulations spécifiques dépassent les seuils d'angle ergonomiques sûrs :
*   **`lgbm_anomaly_neck_hyperflex.txt`** (Seuil : Flexion du cou > 44,5°)
*   **`lgbm_anomaly_shoulder_overext.txt`** (Seuil : Extension/flexion de l'épaule > 96,6°)
*   **`lgbm_anomaly_wrist_strain.txt`** (Seuil : Déviation/flexion du poignet > 36,2°)
*   **`lgbm_anomaly_trunk_torsion.txt`** (Seuil : Rotation du tronc > 64,0°)
*   **`lgbm_anomaly_elbow_hyperext.txt`** (Seuil : Angle interne du coude > 103,0°)

### 3. Prétraitement et scalers
*   **`feature_scaler.pkl`** : StandardScaler mappant le vecteur de 75 caractéristiques dans un espace normalisé pour la prédiction.
*   **`scaler_if.pkl`** : MinMaxScaler utilisé spécifiquement pour normaliser les mesures pour le détecteur d'anomalies Isolation Forest.
*   **`model_metadata.json`** : Dictionnaire JSON central contenant les définitions de modèles, les listes de caractéristiques, les dictionnaires d'encodage, les mesures et les seuils.

---

## 📈 Mesures de performance du modèle

Les modèles de production ont été validés à l'aide d'un schéma de validation croisée **`TimeSeriesSplit`** à 3 plis pour éviter la fuite de données temporelles. Les hyperparamètres ont été ajustés à l'aide d'**Optuna** sur 10 essais d'optimisation.

| Modèle | Mesure | Valeur | État |
|---|---|---|---|
| **Régresseur** | Erreur absolue moyenne (MAE) | **0,0056** | Opérationnel ✅ |
| **Régresseur** | Racine de l'erreur quadratique moyenne (RMSE) | **0,0079** | Opérationnel ✅ |
| **Régresseur** | Coefficient de détermination ($R^2$) | **0,9981** | Opérationnel ✅ |
| **Classificateur de condition** | Précision | **99,60%** | Opérationnel ✅ |
| **Classificateur de condition** | Score F1 (Macro) | **0,9667** | Opérationnel ✅ |
| **Classificateur de gravité** | Précision | **96,95%** | Opérationnel ✅ |
| **Classificateur de gravité** | Score F1 (Macro) | **0,9411** | Opérationnel ✅ |
| **Classificateurs d'anomalies (Moyenne)** | Précision / Score F1 | **99,90% / 0,9906** | Opérationnel ✅ |

---

## 📊 Tracés d'évaluation et de diagnostic
Ce répertoire contient également des tracés d'évaluation visuelle générés par [generate_eval_plots.py](file:///c:/MSD_System/generate_eval_plots.py) et [retrain_v3.py](file:///c:/MSD_System/retrain_v3.py) :
*   `eval_model1_regressor.png` : Courbes Prédit vs Réel et Résiduelles pour la prévision des risques.
*   `eval_model2_confusion.png` : Matrice de confusion de pathologie à 18 classes.
*   `eval_model2_learning.png` & `eval_model2_roc.png` & `eval_model2_pr.png` : Convergence ROC, Précision-Rappel et perte logarithmique.
*   `eval_model3_severity.png` : Mesures d'évaluation de la gravité et matrice de confusion.
*   `eval_model4_bars.png` & `eval_model4_learning.png` & `eval_model4_roc.png` : Mesures d'anomalies articulaires individuelles, ROC et journaux de perte d'entraînement.
*   `shap_classifier.png` & `shap_regressor.png` : Tracés récapitulatifs SHAP soulignant la contribution des 75 caractéristiques aux prédictions de risque et de condition.
