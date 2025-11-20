import os
import pandas as pd

class DataLoader:
    """
    Responsible for loading datasets from the disk.
    """
    def __init__(self, file_path: str):
        """
        Initialize the DataLoader.

        Args:
            file_path (str): The path to the CSV file.
        """
        self.file_path = file_path

    def load_data(self) -> pd.DataFrame:
        """
        Loads data from the CSV file provided during initialization.

        Returns:
            pd.DataFrame: The loaded dataset.
        
        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is empty or corrupt.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"❌ Error: The file at {self.file_path} was not found.")
        
        try:
            df = pd.read_csv(self.file_path)
            print(f"✅ Data loaded successfully! Shape: {df.shape}")
            return df
        except pd.errors.EmptyDataError:
            raise ValueError("❌ Error: The provided CSV file is empty.")
        except Exception as e:
            print(f"❌ Unexpected error while loading data: {e}")
            raise