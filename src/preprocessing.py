import pandas as pd
from typing import List

class DataPreprocessor:
    """
    Handles base data preparation, primarily One-Hot Encoding.
    Note: Scaling and Splitting are delegated to specific model engines
    to support independent experimental setups.
    """
    def __init__(self, df: pd.DataFrame, target_column: str):
        """
        Args:
            df (pd.DataFrame): The dataframe from Feature Engineering.
            target_column (str): The name of the target variable.
        """
        self.df = df.copy()
        self.target_column = target_column

    def prepare_base_dataframe(self) -> pd.DataFrame:
        """
        Performs One-Hot Encoding on categorical features.

        Returns:
            pd.DataFrame: The processed dataframe ready for model-specific scaling.
        """
        print("   ℹ️ Performing Base Data Preparation (One-Hot Encoding)...")
        
        # Columns to encode (including target to replicate reference behavior if needed)
        cat_cols = ['Stage_fear', 'Drained_after_socializing', self.target_column]
        
        # Validate existence
        cols_to_encode = [c for c in cat_cols if c in self.df.columns]
        
        if cols_to_encode:
            self.df = pd.get_dummies(self.df, columns=cols_to_encode, prefix=cols_to_encode)
            
        return self.df