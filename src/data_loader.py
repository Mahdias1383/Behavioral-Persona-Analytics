import pandas as pd
import os

class DataLoader:
    """
    A class responsible for loading data from various sources.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_data(self) -> pd.DataFrame:
        """
        Loads data from a CSV file.
        
        Returns:
            pd.DataFrame: Loaded dataset.
        
        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"❌ Error: The file at {self.file_path} was not found.")
        
        try:
            df = pd.read_csv(self.file_path)
            print(f"✅ Data loaded successfully! Shape: {df.shape}")
            print(df.columns)
            return df
        except Exception as e:
            print(f"❌ Unexpected error while loading data: {e}")
            raise