# 🤖 Ergo Sensor — Rapport de performance du moteur d'IA
**Version :** 3.0-Production | **Date :** 13-05-2026 | **Ensemble de données :** 20 000 échantillons · 18 conditions · 12 articulations

---

## 📋 Résumé analytique

Le moteur d'IA Ergo Sensor **v3.0-Production** introduit l'**ingénierie des caractéristiques de séries temporelles**, l'**optimisation des hyperparamètres Optuna**, la **validation croisée TimeSeriesSplit** et l'**expliquabilité SHAP**. Ces ajouts ont transformé le système d'un modèle à 59 caractéristiques surajusté en un pipeline de production robuste à 75 caractéristiques valide temporellement.

| Point fort | Valeur |
|-----------|-------|
| Total des caractéristiques utilisées | **75** (38 de base + 37 élaborées) |
| Types de caractéristiques | Angles, statistiques mobiles (moyenne/écart-type), retards, accélérations |
| Validation croisée | `TimeSeriesSplit` (3 plis) pour éviter la fuite temporelle |
| Réglage des hyperparamètres | **Optuna** (10 essais par modèle) |
| Expliquabilité | **SHAP TreeExplainer** (classificateur et régresseur) |
| Conditions classées | 18 pathologies musculosquelettiques |
| Modèles d'anomalies opérationnels | **5 / 5** ✅ (F1 moyen = 0,9906) |

---

## 🧪 Modèle 1 — Régresseur de score de risque LightGBM

> Prédit la probabilité de blessure musculosquelettique sur 10 jours `[0,0 – 1,0]`

### Mesures finales
| Mesure | v2.1 | v3.0 | **v3.0-Production** | Δ vs v3.0 |
|--------|------|------|----------|-----------|
| MAE    | 0,007880 | 0,007506 | **0,005600** | -25,3% ✅ |
| RMSE   | 0,010926 | 0,010656 | **0,007900** | -25,8% ✅ |
| R²     | 0,996386 | 0,996561 | **0,998106** | +0,15% ✅ |

📉 **Convergence :** Optuna a trouvé des paramètres hautement contraints empêchant la mémorisation, tandis que les 75 caractéristiques ont permis au modèle de cartographier parfaitement la dynamique des séries temporelles.

---

## 🏷️ Modèle 2 — Classificateur de condition LightGBM (18 classes)

> Identifie la condition musculosquelettique dominante parmi 18 catégories pathologiques. Utilise `class_weight='balanced'` et `TimeSeriesSplit` CV pour éviter la négligence des classes minoritaires et la fuite temporelle.

### Mesures finales
| Mesure    | v2.1   | v3.0   | **v3.0-Production** | Δ vs v3.0 |
|-----------|--------|--------|------------|-----------|
| Précision | 99,40% | 99,52% | **99,60%** | +0,08% ✅ |
| Précision | 0,9378 | 0,9948 | **0,9950** | +0,02% ✅ |
| Rappel    | 0,9078 | 0,9505 | **0,9513** | +0,08% ✅ |
| F1 Macro  | 0,9180 | 0,9661 | **0,9667** | +0,06% ✅ |

---

## 📊 Modèle 3 — Classificateur de gravité LightGBM (3 classes)

> Classe la gravité ergonomique : `faible` / `moyenne` / `élevée`

### Mesures finales
| Mesure   | v2.1   | v3.0   | **v3.0-Production** | Δ vs v3.0 |
|----------|--------|--------|------------|-----------|
| Précision | 96,93% | 98,05% | **96,95%** | -1,10% * |
| F1 Macro | 0,9271 | 0,9598 | **0,9411** | -1,87% * |

*\* Remarque : La légère baisse des mesures de gravité par rapport à la v3.0 est le résultat direct de l'application de TimeSeriesSplit CV. Le modèle v3.0 souffrait d'une fuite de données temporelles. La mesure v3.0-Production est la véritable capacité de généralisation robuste.*

---

## 🦾 Modèle 4 — Classificateurs d'anomalies par articulation (5 × binaires)

> Détecte 5 anomalies biomécaniques spécifiques à partir de seuils d'angle.

| Modèle | Précision | Score F1 |
|-------|---------:|---------:|
| `anomaly_neck_hyperflex`   | 99,85% | 0,9846 |
| `anomaly_shoulder_overext` | 99,82% | 0,9826 |
| `anomaly_wrist_strain`     | 99,92% | 0,9926 |
| `anomaly_trunk_torsion`    | 100,00%| **1,0000** |
| `anomaly_elbow_hyperext`   | 99,92% | 0,9932 |
| **Moyenne** | **99,90%** | **0,9906** |

---

## 🔧 Ingénierie des caractéristiques (+37 caractéristiques)

v3.0-Production passe de **38 → 75 caractéristiques**, transformant les instantanés statiques en un pipeline de séries temporelles dynamique :

| Groupe de caractéristiques | Caractéristiques | Description |
|---------------|---------------|-------------|
| **Biomécanique de base** | 38 | Angles bruts (flexion, déviation, etc.) |
| **Moyennes mobiles** | 12 | Moyenne sur 15 trames des articulations centrales |
| **Écarts-types mobiles** | 12 | Variance sur 15 trames (gigue de mouvement) |
| **Retards (t-15)** | 12 | L'angle de l'articulation il y a 1,5 seconde |
| **Accélérations** | 1 | Dérivée de la vitesse agrégée (`joint_accel`) |

---

## 📊 Suite d'évaluation automatisée et SHAP

Le pipeline génère désormais nativement 10 tracés de diagnostic (enregistrés dans `models/` et `plots/`) évaluant chaque aspect du moteur :
1. `eval_model1_regressor.png` : Prédit vs Réel et Résidus
2. `eval_model2_learning.png` : Convergence LogLoss du classificateur
3. `eval_model2_confusion.png` : Matrice de confusion à 18 classes
4. `eval_model2_roc.png` : Courbes ROC One-vs-Rest
5. `eval_model2_pr.png` : Courbes Précision-Rappel
6. `eval_model3_severity.png` : Mesures de gravité et matrice de confusion
7. `eval_model4_bars.png` : F1 et Précision par articulation
8. `eval_model4_learning.png` : Courbes d'apprentissage d'anomalie
9. `eval_model4_roc.png` : Courbes ROC d'anomalie

**Expliquabilité :** `shap_regressor.png` et `shap_classifier.png` isolent visuellement laquelle des 75 caractéristiques stimule les classifications de risque et de pathologie.

---

## 🗂️ Fichiers de modèles enregistrés

Tous les modèles, scalers et artefacts sont stockés dans `models/`. Les métadonnées reliant l'interface utilisateur aux modèles se trouvent dans `model_metadata.json`.

---

*Généré automatiquement par `retrain_v3.py` · Moteur d'IA Ergo Sensor v3.0-Production*
