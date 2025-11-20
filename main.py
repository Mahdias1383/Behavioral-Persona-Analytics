import sys
import os
import traceback

# Ensure src module can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.preprocessing import DataPreprocessor
from src.eda import EDAReport
from src.ml_engine import MLEngine
from src.dl_engine import DeepLearningEngine

# Configuration
DATA_PATH = 'data/personality_dataset.csv'
TARGET_COLUMN = 'Personality'

def main():
    print("🚀 Starting Final Project Pipeline (Target: 100% Accuracy)...")
    
    # 1. Load
    loader = DataLoader(DATA_PATH)
    try:
        df = loader.load_data()
    except Exception:
        return

    # 2. Feature Engineering
    print("\n--- Step 2: Feature Engineering ---")
    try:
        fe = FeatureEngineer(df)
        df = fe.apply_engineering()
    except Exception as e:
        print(f"❌ Feature Eng Error: {e}")
        traceback.print_exc()
        return

    # 3. Preprocessing (Global)
    print("\n--- Step 3: Preprocessing ---")
    preprocessor = DataPreprocessor(df, target_column=TARGET_COLUMN)
    
    # This runs encoding/scaling and returns X/y for classic models (Clean X)
    # Note: X_train/y_train here are from the global split, but ML/DL engines 
    # might do their own local splits/scaling if configured to do so.
    # In our latest strategy, ml_engine does local prep.
    X_train, X_test, y_train, y_test = preprocessor.process_and_split()
    
    # Get the FULL dataframe (with potential leakage cols) for independent pipelines
    df_full = preprocessor.prepare_base_dataframe()
    
    target_names = ['Extrovert', 'Introvert']

    # 4. Classic ML Models
    print("\n--- Step 4: Classic ML Models ---")
    ml_engine = MLEngine()
    
    # FIX: Call the correct method name 'run_all_classic_models'
    # We pass df_full and target_column because the engine handles prep internally now.
    ml_engine.run_all_classic_models(df_full, TARGET_COLUMN)

    # 5. Deep Learning (ANN - Exact Replication)
    print("\n--- Step 5: Deep Learning (ANN) ---")
    try:
        # Input dim will be determined dynamically inside
        dl_engine = DeepLearningEngine() 
        
        # Pass the FULL dataframe (df_full) which matches the csv analysis
        dl_engine.execute_ann_pipeline(df_full, TARGET_COLUMN)
    except Exception as e:
        print(f"❌ ANN Error: {e}")
        traceback.print_exc()

    print("\n✨ Pipeline Execution Finished! Check 'reports/evaluation' for results.")

if __name__ == "__main__":
    main()