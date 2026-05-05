"""
=============================================================================
PIPELINE COMPLET - EXÉCUTION END-TO-END
=============================================================================
Auteur: Ingénieur IA Senior
Date: 2025
Version: 1.0

✅ Exécute tout le pipeline:
   1. Génération du dataset
   2. Entraînement des modèles
   3. Tests et validation
   4. Inférence et explainability

À exécuter une seule fois:
    python run_complete_pipeline.py
=============================================================================
"""

import os
import subprocess
import sys
import time
from datetime import datetime

# ============================================================================
# SETUP
# ============================================================================

def setup_directories():
    """
    Crée les répertoires nécessaires
    """
    dirs = ['models', 'plots', 'data', 'logs']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"✅ Dossier créé: {d}/")

def install_dependencies():
    """
    Installe les dépendances Python
    """
    print("\n[1/5] Installation des dépendances...\n")
    
    dependencies = [
        'pandas',
        'numpy',
        'scikit-learn',
        'lightgbm',
        'tensorflow',
        'shap',
        'matplotlib',
        'seaborn',
        'joblib'
    ]
    
    for dep in dependencies:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', dep])
            print(f"   ✅ {dep}")
        except subprocess.CalledProcessError as e:
            print(f"   ⚠️  {dep} - Erreur lors de l'installation (peut être déjà installé)")

# ============================================================================
# EXÉCUTION COMPLÈTE
# ============================================================================

def run_complete_pipeline():
    """
    Exécute le pipeline complet
    """
    
    print("\n" + "="*80)
    print("PIPELINE COMPLET - TMS PREDICTION SYSTEM")
    print("="*80 + "\n")
    
    start_time = datetime.now()
    
    # Setup
    print("SETUP")
    print("-" * 80)
    setup_directories()
    
    # Installation des dépendances
    install_dependencies()
    
    print("\n" + "="*80)
    print("ÉTAPE 1: GÉNÉRATION DU DATASET")
    print("="*80 + "\n")
    
    # Exécuter la génération du dataset
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("generate", "01_generate_dataset.py")
        generate_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generate_module)
        
        # Exécuter la fonction principale
        df, mappings, scaler = generate_module.generate_complete_dataset()
        print("\n✅ Dataset généré avec succès")
        
    except Exception as e:
        print(f"\n❌ Erreur lors de la génération du dataset: {e}")
        return False
    
    time.sleep(2)
    
    print("\n" + "="*80)
    print("ÉTAPE 2: ENTRAÎNEMENT DES MODÈLES")
    print("="*80 + "\n")
    
    # Exécuter l'entraînement
    try:
        spec = importlib.util.spec_from_file_location("training", "02_train_models.py")
        training_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(training_module)
        
        # Charger les données
        df, mappings, scaler = training_module.load_dataset('dataset_TMS_enriched.csv')
        
        # Entraîner tous les modèles
        results, preprocessor, feature_scaler, lgb_reg, lgb_cls, cnn, lstm, hybrid = \
            training_module.train_all_models(df, mappings, scaler)
        
        # Rapport final
        training_module.generate_final_report(results)
        
        print("\n✅ Tous les modèles entraînés avec succès")
        
    except Exception as e:
        print(f"\n❌ Erreur lors de l'entraînement: {e}")
        return False
    
    time.sleep(2)
    
    print("\n" + "="*80)
    print("ÉTAPE 3: INFÉRENCE ET VALIDATION")
    print("="*80 + "\n")
    
    # Exécuter l'inférence
    try:
        spec = importlib.util.spec_from_file_location("inference", "03_inference.py")
        inference_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(inference_module)
        
        print("✅ Module d'inférence chargé\n")
        
        # Exemple 1: Prédiction simple
        print("\n[Exemple 1] Prédiction simple...")
        predictions_1 = inference_module.example_single_prediction()
        
        # Exemple 2: Batch
        print("\n[Exemple 2] Batch prediction...")
        predictions_2 = inference_module.example_batch_prediction()
        
        # Exemple 3: Avec SHAP
        print("\n[Exemple 3] Prédiction avec explainability...")
        predictions_3, feat_imp = inference_module.example_with_explainability()
        
        print("\n✅ Inférence et validation réussies")
        
    except Exception as e:
        print(f"\n❌ Erreur lors de l'inférence: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Résumé final
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds() / 60
    
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLET EXÉCUTÉ AVEC SUCCÈS")
    print("="*80 + "\n")
    
    print(f"⏱️  Temps total: {elapsed:.1f} minutes\n")
    
    print("📁 Fichiers générés:")
    print("   Dataset:     dataset_TMS_enriched.csv")
    print("   Modèles:     models/")
    print("   Plots:       plots/")
    print("   Métadonnées: condition_mappings.json, risk_scaler.pkl\n")
    
    print("🚀 Prochaines étapes:")
    print("   1. Utiliser 03_inference.py pour faire des prédictions")
    print("   2. Intégrer l'API dans une application web")
    print("   3. Monitorer les performances en production")
    print("   4. Re-entraîner les modèles régulièrement\n")
    
    return True

# ============================================================================
# TESTS UNITAIRES BASIQUES
# ============================================================================

def run_basic_tests():
    """
    Exécute quelques tests basiques de validation
    """
    
    print("\n" + "="*80)
    print("TESTS DE VALIDATION")
    print("="*80 + "\n")
    
    try:
        # Test 1: Vérifier que le dataset existe
        print("[Test 1] Vérification du dataset...")
        import pandas as pd
        df = pd.read_csv('dataset_TMS_enriched.csv')
        assert len(df) > 0, "Dataset vide"
        assert 'risk_score' in df.columns, "risk_score manquant"
        assert 'main_condition' in df.columns, "main_condition manquant"
        print(f"✅ Dataset valide ({len(df)} rows, {len(df.columns)} columns)\n")
        
        # Test 2: Vérifier que les modèles existent
        print("[Test 2] Vérification des modèles...")
        import os
        model_files = [
            'models/lgb_regressor.txt',
            'models/lgb_classifier.txt',
            'models/cnn_model.h5',
            'models/lstm_model.h5',
            'models/hybrid_cnn.h5',
            'models/hybrid_lgb.txt'
        ]
        for mf in model_files:
            assert os.path.exists(mf), f"{mf} manquant"
        print(f"✅ Tous les modèles présents\n")
        
        # Test 3: Vérifier les métadonnées
        print("[Test 3] Vérification des métadonnées...")
        import json
        with open('condition_mappings.json', 'r') as f:
            mappings = json.load(f)
        assert 'condition_to_code' in mappings, "condition_to_code manquant"
        print(f"✅ Métadonnées valides\n")
        
        # Test 4: Test d'inférence simple
        print("[Test 4] Test d'inférence simple...")
        from inference import InferenceAPI
        api = InferenceAPI()
        result = api.predict({
            'neck': 25.0, 'trunk': 30.0, 'shoulder': 50.0, 'wrist': 15.0,
            'neck_vel': 1.0, 'trunk_vel': 1.5, 'shoulder_vel': 1.0, 'wrist_vel': 0.5,
            'neck_duration': 5, 'trunk_duration': 8, 'shoulder_duration': 4, 'wrist_duration': 2,
            'neck_freq': 1, 'trunk_freq': 2
        })
        assert 'risk_score' in result, "risk_score manquant"
        print(f"✅ Inférence fonctionnelle (score: {result['risk_score'][0]:.4f})\n")
        
        print("✅ TOUS LES TESTS PASSÉS\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Test échoué: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "PIPELINE COMPLET - TMS PREDICTION SYSTEM" + " "*19 + "║")
    print("║" + " "*25 + "Ingénieur IA Senior - 2025" + " "*25 + "║")
    print("╚" + "="*78 + "╝")
    
    # Exécuter le pipeline complet
    success = run_complete_pipeline()
    
    if success:
        # Exécuter les tests
        print("\n")
        tests_pass = run_basic_tests()
        
        if tests_pass:
            print("\n" + "="*80)
            print("🎉 PIPELINE COMPLÈTEMENT OPÉRATIONNEL")
            print("="*80 + "\n")
            
            print("📚 Documentation:")
            print("   01_generate_dataset.py  → Génération du dataset")
            print("   02_train_models.py      → Entraînement des modèles")
            print("   03_inference.py         → Prédictions et explainability\n")
            
            print("🚀 Pour utiliser:")
            print("   from inference import InferenceAPI")
            print("   api = InferenceAPI()")
            print("   results = api.predict(data_dict, explain=True)\n")
        else:
            print("\n⚠️  Pipeline généré mais certains tests ont échoué")
            sys.exit(1)
    else:
        print("\n❌ Pipeline échoué")
        sys.exit(1)
