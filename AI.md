# 🧠 Ergo Sensor — Intelligence artificielle et modélisation biomécanique

Ce document fournit une plongée profonde dans le noyau d'Intelligence Artificielle du système **Ergo Sensor**. Il explique la justification scientifique, les architectures de modèles et le pipeline de données utilisé pour prévenir les troubles musculosquelettiques (TMS).

---

## 🔬 Pourquoi ce choix de modèle ?

Nous avons sélectionné une approche d'**Apprentissage d'ensemble** basée sur **LightGBM** pour plusieurs raisons techniques :

1.  **Séquences temporelles** : Ergo Sensor v3.0-Production s'appuie sur l'ingénierie des caractéristiques de séries temporelles (fenêtres mobiles, retards) et la validation croisée `TimeSeriesSplit` pour capturer l'historique postural dynamique plutôt que des instantanés statiques.
2.  **Optuna HPO** : L'optimisation des hyperparamètres (Optuna) garantit que les modèles se généralisent aux travailleurs invisibles plutôt que de mémoriser les données d'entraînement.
3.  **Gestion du déséquilibre** : Les anomalies posturales sont rares. Le `class_weight='balanced'` de LightGBM nous permet de détecter efficacement les blessures rares.
4.  **Gestion des inconnus** : Bien qu'il s'agisse principalement d'un système supervisé, l'approche est structurée pour gérer également diverses anomalies posturales inconnues.

---

## 📊 L'ensemble de données : `dataset_TMS_enriched.csv`

Le modèle a été entraîné sur un ensemble de données de haute qualité spécifiquement conçu pour la recherche sur les troubles musculosquelettiques (TMS) :

- **Volume** : ~20 000 points de données collectés à 10 Hz.
- **Caractéristiques d'entrée (75)** : 
    - 12 angles articulaires bruts (norme clinique).
    - 24 statistiques temporelles (moyennes et écarts-types mobiles).
    - 12 caractéristiques de retard (historique postural).
    - 14 caractéristiques dynamiques et proxys (vitesse, énergie).
    - 5 deltas d'asymétrie bilatérale.
    - 2 scores de charge composite.
    - 6 superpositions d'angles bruts, drapeaux de posture et accélérations.
- **Classes cibles** : 
    - **Continu** : Score de risque (0,0 à 1,0).
    - **Multi-classes** : 18 catégories de conditions (pathologies TMS).
    - **Gravité** : Faible, Modérée, Élevée.

---

## ⚙️ Développement et entraînement du modèle

Le pipeline d'entraînement suit un flux de travail rigoureux en science des données :

1.  **Ingénierie des caractéristiques** : Extraction du vecteur biomécanique de **75 caractéristiques** via `feature_extractor.py`.
2.  **Poids équilibrés** : Utilisation d'un entraînement pondéré par classe pour compenser le déséquilibre sévère de l'ensemble de données.
3.  **Division temporelle** : Validation croisée `TimeSeriesSplit` à 3 plis pour garantir une fuite de données temporelles nulle.
4.  **Optimisation** : Réglage des hyperparamètres à l'aide de l'optimisation bayésienne (Optuna).

---

## 📈 Résultats de performance (v3.0-Production)

L'IA Ergo Sensor obtient des résultats de pointe pour l'évaluation ergonomique en temps réel :

| Mesure | Résultat | Interprétation |
|---|---|---|
| **Score R² (Risque)** | **0,9981** | Variance quasi parfaite expliquée dans la prévision des risques de blessures. |
| **Précision (Cond)** | **99,60%** | Classification exceptionnelle à travers 18 pathologies TMS. |
| **Score F1 (Gravité)** | **0,9411** | Généralisation robuste sur la gravité malgré la validation croisée temporelle. |
| **Score F1 (Anomalie)** | **0,9906** | F1 moyen à travers 5 classificateurs d'anomalies par articulation distincts. |

---

## 🔍 Expliquabilité avec SHAP

L'une des fonctionnalités les plus puissantes d'Ergo Sensor est **SHAP (SHapley Additive exPlanations)**. 

Lorsqu'un "risque élevé" est détecté, l'IA ne se contente pas de donner un chiffre. Elle utilise **TreeExplainer** pour calculer la contribution de chacune des 75 caractéristiques. 
- **Exemple** : "Le risque est de 85 %. **Cause principale** : Flexion soutenue du tronc (contribution de 42 %) et retard de l'épaule."
- **Bénéfice** : Cela permet aux cliniciens de fournir un retour spécifique (ex: "Ajustez la hauteur de votre chaise pour abaisser votre épaule droite").

---

## 🔄 Pipeline de données d'IA

1.  **Ingestion** : Quaternions bruts des capteurs ESP32 via Firebase.
2.  **Traitement** : `angle_math.py` convertit l'orientation en 12 angles articulaires cliniques.
3.  **Extraction** : `feature_extractor.py` calcule le vecteur de série temporelle dynamique de 75 caractéristiques.
4.  **Inférence** : `ai_engine.py` exécute l'ensemble de 5 modèles.
5.  **Visualisation** : Tableau de bord en direct, squelette 3D et rapports PDF avec **courbes de probabilité d'anomalies**.
