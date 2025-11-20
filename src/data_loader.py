import os
import pandas as pd

class DataLoader:
    """
    Handles the ingestion of data from various sources (currently CSV).
    Ensures data integrity upon loading.
    """
    def __init__(self, file_path: str):
        """
        Initializes the DataLoader.

        Args:
            file_path (str): Path to the CSV dataset file.
        """
        self.file_path = file_path

    def load_data(self) -> pd.DataFrame:
        """
        Loads the dataset from the specified file path.

        Returns:
            pd.DataFrame: The loaded pandas DataFrame.
        
        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is empty or corrupt.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"❌ Critical Error: Dataset not found at {self.file_path}")
        
        try:
            df = pd.read_csv(self.file_path)
            if df.empty:
                raise ValueError("The dataset is empty.")
            
            print(f"✅ Data loaded successfully! Shape: {df.shape}")
            return df
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise