import pandas as pd
import numpy as np

class FeatureEngineer:
    """
    Handles feature engineering tasks including imputation of missing values
    and creation of derived features based on domain knowledge.
    """
    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df (pd.DataFrame): The raw dataframe.
        """
        self.df = df.copy()

    def apply_engineering(self) -> pd.DataFrame:
        """
        Applies the feature engineering pipeline.
        
        Steps:
        1. Mean imputation for numerical columns.
        2. Forward-fill imputation for categorical columns.
        3. Manual numeric mapping for specific categorical features.
        4. Target encoding (0/1).
        5. Creation of derived metrics (Social Score, Online Ratio).

        Returns:
            pd.DataFrame: The dataframe with engineered features.
        """
        print("   🛠️ Applying Feature Engineering...")
        
        # 1. Impute Numerical Columns (Mean)
        num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                    'Friends_circle_size', 'Post_frequency']
        for col in num_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(self.df[col].mean())

        # 2. Impute Categorical Columns (ffill with fallback)
        cat_cols = ['Stage_fear', 'Drained_after_socializing']
        for col in cat_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].ffill()
                # Fallback for initial rows
                if self.df[col].isnull().any():
                    self.df[col] = self.df[col].fillna(method='bfill')
                    # Ultimate fallback
                    if self.df[col].isnull().any():
                        self.df[col] = self.df[col].fillna(self.df[col].mode()[0])

        # 3. Manual Mapping
        if 'Stage_fear' in self.df.columns:
            self.df['Stage_fear_num'] = self.df['Stage_fear'].map({'Yes': 1, 'No': 0})
            
        if 'Drained_after_socializing' in self.df.columns:
            self.df['Drained_after_socializing_num'] = self.df['Drained_after_socializing'].map({'Yes': 1, 'No': 0})

        # 4. Encode Target (Extrovert=0, Introvert=1)
        if 'Personality' in self.df.columns:
            self.df['Personality_num'] = self.df['Personality'].map({'Extrovert': 0, 'Introvert': 1})

        # 5. Derived Features
        self.df['Social_Interaction_Score'] = (
            self.df['Social_event_attendance'] + 
            self.df['Friends_circle_size'] + 
            self.df['Post_frequency']
        )

        # Add epsilon to avoid division by zero
        self.df['Online_Offline_Ratio'] = (
            self.df['Post_frequency'] / (self.df['Going_outside'] + 1.0)
        )

        # Final cleanup of any NaNs created during derivation
        new_cols = ['Social_Interaction_Score', 'Online_Offline_Ratio', 'Stage_fear_num', 'Drained_after_socializing_num']
        for col in new_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(0)

        return self.df