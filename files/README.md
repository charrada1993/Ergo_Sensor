# ⚠️ Scripts de référence d'IA hérités (v1.0)

Ce répertoire contient les scripts de référence originaux de la version v1.0, la documentation et le code source du prototype initial de l'IA.

> [!WARNING]
> Les scripts de ce répertoire sont conservés **uniquement pour référence historique et compatibilité ascendante**. Ils utilisent un modèle d'instantané hérité à 14 caractéristiques et des architectures de réseaux neuronaux de base (CNN 1D, LSTM et modèles hybrides).
>
> Pour la base de code active de qualité production (v3.0-Production), reportez-vous aux fichiers du **répertoire racine** du dépôt.

---

## 📂 Manifeste des fichiers hérités
*   **`01_generate_dataset.py`** : Script hérité pour la génération d'ensembles de données biomécaniques synthétiques à 14 caractéristiques.
*   **`02_train_models.py`** : Script hérité pour l'entraînement de cinq modèles de réseaux neuronaux prototypes (CNN, LSTM, hybride CNN+LGB) utilisant une division aléatoire de base.
*   **`03_inference.py`** : Script d'inférence unique hérité pour vérifier les prédictions du prototype.
*   **`run_complete_pipeline.py`** : Script coordinateur hérité pour exécuter la génération et l'entraînement de l'ensemble de données de bout en bout v1.0.
*   **`TECHNICAL_SUMMARY.txt`** : Notes de documentation héritées décrivant les architectures des réseaux neuronaux v1.0.

---

## 🔄 Différences : Héritage (v1.0) vs Production active (v3.0)

| Caractéristique | Héritage v1.0 (`files/`) | Production active v3.0 (Racine `c:/MSD_System/`) |
|---|---|---|
| **Caractéristiques utilisées** | **14 caractéristiques de base** (instantanés posturaux statiques) | **75 caractéristiques élaborées** (statistiques mobiles, indices de retard, asymétries bilatérales, vitesses, charges articulaires composites et accélérations) |
| **Architectures de modèles** | CNN 1D, LSTM, Hybride (CNN+LGB), LightGBM | Ensemble multi-modèle hautement optimisé de régresseurs LightGBM, classificateurs de condition, classificateurs de gravité et 5 anomalies articulaires binaires |
| **Validation croisée** | Division aléatoire standard (souffre de fuite de données temporelles) | **`TimeSeriesSplit` (3 plis)** (assure la validité temporelle et empêche la fuite de données) |
| **Hyperparamètres** | Paramètres codés en dur/manuels | **Optimisation Optuna** (recherches d'hyperparamètres automatisées en 10 essais) |
| **Moteurs ergonomiques** | Angles simples | Moteurs programmatiques bilatéraux **RULA & REBA** avec étalonnage dynamique |
| **Expliquabilité** | Graphiques SHAP de base | **SHAP TreeExplainer** natif entièrement intégré au tableau de bord clinicien en direct |
| **Visualisations** | Aucune | Miroir squelette **Jumeau numérique 3D Three.js** en temps réel |

---

## 🚀 Comment exécuter le pipeline de production actuel

Pour entraîner, optimiser et servir le système v3.0-Production actif, utilisez les fichiers suivants situés dans le répertoire racine :
1.  **Entraînement du modèle** : Exécutez [retrain_v3.py](file:///c:/MSD_System/retrain_v3.py) pour exécuter l'HPO Optuna, le CV TimeSeriesSplit et l'ingénierie des caractéristiques.
2.  **Traitement de la télémétrie** : Consultez [angle_math.py](file:///c:/MSD_System/angle_math.py) pour les conversions quaternion vers articulation et [feature_extractor.py](file:///c:/MSD_System/feature_extractor.py) pour l'extraction de la fenêtre de 75 caractéristiques.
3.  **Tableau de bord en temps réel et serveur backend** : Démarrez l'application Flask en exécutant [app.py](file:///c:/MSD_System/app.py).
