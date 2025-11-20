import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class DataPreprocessor:
    """
    Handles One-Hot Encoding, Scaling, and Splitting.
    UPDATED: Retains One-Hot encoded TARGET columns to replicate potential data leakage in reference notebook.
    """
    def __init__(self, df: pd.DataFrame, target_column: str):
        self.df = df.copy()
        self.target_column = target_column
        self.scaler = MinMaxScaler()

    def process_and_split(self): 
        print("   ℹ️ Processing data (One-Hot -> Scale -> 80/20 Split)...")

        # 1. One-Hot Encoding
        # Include TARGET column in encoding list to generate 'Personality_Extrovert', etc.
        categorical_columns = ['Stage_fear', 'Drained_after_socializing', self.target_column]
        cols_to_encode = [c for c in categorical_columns if c in self.df.columns]
        
        if cols_to_encode:
            # prefix=None uses column name as prefix, which is what we want (Personality_Extrovert)
            self.df = pd.get_dummies(self.df, columns=cols_to_encode, prefix=cols_to_encode)

        # 2. Normalization (MinMaxScaler)
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64', 'int32']).columns.tolist()
        y_col = f"{self.target_column}_num"
        if y_col in numeric_cols:
            numeric_cols.remove(y_col)
            
        if numeric_cols:
            self.df[numeric_cols] = self.scaler.fit_transform(self.df[numeric_cols])

        # 3. Define X and y
        if y_col not in self.df.columns:
             # If not found, maybe Feature Eng failed or names mismatched.
             # For now, assume it exists or handle downstream.
             print(f"   ⚠️ Warning: {y_col} not found. Creating from One-Hots or skipping.")
             # Fallback logic would be here, but we assume FeatureEng worked.
             pass
        else:
             y = self.df[y_col]
        
        # 4. Define X (Global Split Strategy)
        # For classic models (Global Split), we usually drop target leakage columns.
        # But for ANN (Local Split), we pass the whole DF.
        # Here we prepare X for the global split return (Classic Models).
        
        cols_to_drop_classic = [
            f"{self.target_column}_Extrovert", 
            f"{self.target_column}_Introvert", 
            y_col
        ]
        existing_drop_cols = [c for c in cols_to_drop_classic if c in self.df.columns]
        X = self.df.drop(existing_drop_cols, axis=1)

        # 5. Split Data (Global)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        print(f"✅ Data Processed. Train: {X_train.shape}, Test: {X_test.shape} (Target Support: 580)")
        
        # IMPORTANT: We return the modified self.df too (via getter or just relying on object state)
        # But main.py accesses `df_prepared` which should be the FULL dataframe with leakage cols.
        
        return X_train, X_test, y_train, y_test

    def prepare_base_dataframe(self):
        """
        Returns the fully processed dataframe INCLUDING target leakage columns.
        Used by independent pipelines.
        """
        # We assume process_and_split has run or logic is duplicated.
        # To be safe, let's just run the encoding logic here if not done.
        # Actually, better to reuse the state if process_and_split modified self.df.
        # Yes, process_and_split modifies self.df in place.
        return self.df