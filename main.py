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
    print("🚀 Starting Project Pipeline (Independent Model Strategy)...\n")
    
    # 1. Load
    if not os.path.exists(DATA_PATH):
        print(f"❌ Dataset not found at {DATA_PATH}")
        return
    loader = DataLoader(DATA_PATH)
    df = loader.load_data()

    # 2. EDA
    print("\n--- Step 2: EDA ---")
    try:
        eda = EDAReport(df)
        eda.generate_inspection_report()
        print("   ✅ EDA generated.")
    except Exception as e:
        print(f"⚠️ EDA Warning: {e}")

    # 3. Feature Engineering
    print("\n--- Step 3: Feature Engineering ---")
    try:
        fe = FeatureEngineer(df)
        df = fe.apply_engineering()
    except Exception as e:
        print(f"❌ Feature Eng Error: {e}")
        return

    # 4. Base Preprocessing (Just One-Hot, no global split/scale)
    print("\n--- Step 4: Base Data Preparation ---")
    preprocessor = DataPreprocessor(df, target_column=TARGET_COLUMN)
    df_prepared = preprocessor.prepare_base_dataframe()

    # 5. Classic ML Models (Each does its own split/scale locally)
    print("\n--- Step 5: Classic ML Models (Local Pipelines) ---")
    ml_engine = MLEngine()
    ml_engine.run_all_classic_models(df_prepared, TARGET_COLUMN)

    # 6. Deep Learning (ANN - Independent Pipeline)
    print("\n--- Step 6: Deep Learning (ANN - Independent Pipeline) ---")
    try:
        dl_engine = DeepLearningEngine()
        # Passing the FULL dataframe. The engine will split and scale it internally.
        dl_engine.execute_ann_pipeline(df_prepared, TARGET_COLUMN)
    except Exception as e:
        print(f"❌ ANN Error: {e}")
        traceback.print_exc()

    print("\n✨ Pipeline Finished!")

if __name__ == "__main__":
    main()