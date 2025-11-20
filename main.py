import sys
import os
import traceback

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.preprocessing import DataPreprocessor
from src.eda import EDAReport
from src.ml_engine import MLEngine
from src.dl_engine import DeepLearningEngine

DATA_PATH = 'data/personality_dataset.csv'
TARGET_COLUMN = 'Personality'

def main():
    print("🚀 Starting Behavioral Persona Analytics Pipeline (Production Mode)...\n")
    
    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"❌ Critical Error: Dataset not found at {DATA_PATH}")
        return
    
    loader = DataLoader(DATA_PATH)
    try:
        df = loader.load_data()
    except Exception as e:
        print(f"❌ Loading Failed: {e}")
        return

    # 2. Initial EDA
    print("\n--- Step 2: Initial Exploratory Data Analysis ---")
    try:
        eda = EDAReport(df)
        eda.generate_inspection_report()
        eda.plot_anomalies_boxplot()
        eda.plot_categorical_anomalies()
        eda.plot_correlation_matrix(title_suffix=" (Raw Data)")
        print("   ✅ Initial EDA completed.")
    except Exception as e:
        print(f"⚠️ EDA Warning: {e}")
        traceback.print_exc()

    # 3. Feature Engineering
    print("\n--- Step 3: Feature Engineering ---")
    try:
        fe = FeatureEngineer(df)
        df = fe.apply_engineering()
    except Exception as e:
        print(f"❌ Feature Engineering Error: {e}")
        return

    # 4. Preprocessing (Base One-Hot)
    print("\n--- Step 4: Base Preprocessing ---")
    preprocessor = DataPreprocessor(df, target_column=TARGET_COLUMN)
    df_full = preprocessor.prepare_base_dataframe()

    # 5. Classic ML Models (Independent Pipelines)
    print("\n--- Step 5: Classic ML Models ---")
    try:
        ml_engine = MLEngine()
        ml_engine.run_all_classic_models(df_full, TARGET_COLUMN)
    except Exception as e:
        print(f"❌ ML Engine Error: {e}")
        traceback.print_exc()

    # 6. Deep Learning (Replicated Pipeline)
    print("\n--- Step 6: Deep Learning (ANN) ---")
    try:
        dl_engine = DeepLearningEngine()
        dl_engine.execute_ann_pipeline(df_full, TARGET_COLUMN)
    except Exception as e:
        print(f"❌ DL Engine Error: {e}")
        traceback.print_exc()

    print("\n✨ Pipeline Execution Finished Successfully!")
    print("   📂 Check 'reports/' folder for interactive HTML visualizations.")

if __name__ == "__main__":
    main()