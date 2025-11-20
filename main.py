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
    print("🚀 Starting Final Project Pipeline ...")
    
    # 1. Load
    loader = DataLoader(DATA_PATH)
    try:
        df = loader.load_data()
    except Exception:
        return

    # 2. Initial EDA
    print("\n--- Step 2: Initial Exploratory Data Analysis ---")
    try:
        eda = EDAReport(df)
        # FIX: Explicitly calling the generation methods!
        eda.generate_inspection_report()
        eda.plot_anomalies_boxplot()
        eda.plot_categorical_anomalies()
        eda.plot_correlation_matrix(title_suffix=" (Before Feature Eng)")
        print("   ✅ Initial EDA reports generated.")
    except Exception as e:
        print(f"⚠️ EDA Warning: {e}")
        traceback.print_exc()

    # 3. Feature Engineering
    print("\n--- Step 3: Feature Engineering ---")
    try:
        fe = FeatureEngineer(df)
        df = fe.apply_engineering()
    except Exception as e:
        print(f"❌ Feature Eng Error: {e}")
        traceback.print_exc()
        return

    # 4. Secondary EDA (After Feature Eng)
    print("\n--- Step 4: Secondary EDA ---")
    try:
        # Re-initialize with new df to plot new features
        eda = EDAReport(df)
        eda.plot_correlation_matrix(title_suffix=" (With Derived Features)")
        print("   ✅ Secondary EDA reports generated.")
    except Exception as e:
        print(f"⚠️ Secondary EDA Warning: {e}")

    # 5. Preprocessing
    print("\n--- Step 5: Preprocessing ---")
    preprocessor = DataPreprocessor(df, target_column=TARGET_COLUMN)
    
    # Global Split for Classic Models
    X_train, X_test, y_train, y_test = preprocessor.process_and_split()
    
    # Full DF for Independent Pipelines
    df_full = preprocessor.prepare_base_dataframe()
    
    target_names = ['Extrovert', 'Introvert']

    # 6. Classic ML Models
    print("\n--- Step 6: Classic ML Models ---")
    ml_engine = MLEngine()
    ml_engine.run_all_classic_models(df_full, TARGET_COLUMN)

    # 7. Deep Learning (ANN)
    print("\n--- Step 7: Deep Learning (ANN) ---")
    try:
        dl_engine = DeepLearningEngine() 
        dl_engine.execute_ann_pipeline(df_full, TARGET_COLUMN)
    except Exception as e:
        print(f"❌ ANN Error: {e}")
        traceback.print_exc()

    print("\n✨ Pipeline Execution Finished! Check 'reports/' for all outputs.")

if __name__ == "__main__":
    main()