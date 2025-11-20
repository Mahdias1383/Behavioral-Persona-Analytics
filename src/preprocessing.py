import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class DataPreprocessor:
    """
    Performs BASE data preparation (One-Hot Encoding).
    Scaling and Splitting are now delegated to individual model engines
    to match the cell-by-cell logic of the reference notebook.
    """
    def __init__(self, df: pd.DataFrame, target_column: str):
        self.df = df.copy()
        self.target_column = target_column

    def prepare_base_dataframe(self):
        print("   ℹ️ Performing Base Data Preparation (One-Hot Encoding)...")
        
        # One-Hot Encoding Categoricals
        cat_cols = ['Stage_fear', 'Drained_after_socializing']
        cols_to_encode = [c for c in cat_cols if c in self.df.columns]
        
        if cols_to_encode:
            self.df = pd.get_dummies(self.df, columns=cols_to_encode, prefix=cols_to_encode)
            
        # Note: We DO NOT scale or split here anymore.
        # We pass the enriched DataFrame to the ML/DL engines.
        
        return self.df