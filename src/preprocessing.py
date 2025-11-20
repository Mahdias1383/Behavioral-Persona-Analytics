import pandas as pd

class DataPreprocessor:
    """
    Handles base data preparation, primarily One-Hot Encoding.
    
    NOTE: Actual Scaling (Standard/MinMax) and Splitting are delegated 
    to the respective Model Engines (MLEngine, DeepLearningEngine) to ensure 
    independent experimental conditions matching the reference study.
    """
    def __init__(self, df: pd.DataFrame, target_column: str):
        self.df = df.copy()
        self.target_column = target_column

    def prepare_base_dataframe(self) -> pd.DataFrame:
        """
        Performs One-Hot Encoding on categorical features.
        
        Returns:
            pd.DataFrame: Dataframe with One-Hot encoded columns added.
        """
        print("   ℹ️ Performing Base Data Preparation (One-Hot Encoding)...")
        
        # Columns to encode (including target to replicate reference behavior)
        cat_cols = ['Stage_fear', 'Drained_after_socializing', self.target_column]
        
        # Verify existence before encoding
        cols_to_encode = [c for c in cat_cols if c in self.df.columns]
        
        if cols_to_encode:
            # drop_first=False to keep all categories (Yes/No, Introvert/Extrovert)
            self.df = pd.get_dummies(self.df, columns=cols_to_encode, prefix=cols_to_encode)
            
        return self.df