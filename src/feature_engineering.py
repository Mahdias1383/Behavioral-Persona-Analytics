import pandas as pd
import numpy as np

class FeatureEngineer:
    """
    Responsible for feature engineering, including:
    1. Missing value imputation (Mean/Mode/FFill).
    2. Manual encoding of specific categorical features.
    3. Creation of derived features (Social Scores).
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def apply_engineering(self) -> pd.DataFrame:
        """
        Executes the feature engineering pipeline.
        
        Returns:
            pd.DataFrame: The engineered dataframe ready for preprocessing.
        """
        print("   🛠️ Applying Feature Engineering...")
        
        # 1. Numerical Imputation (Mean)
        num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                    'Friends_circle_size', 'Post_frequency']
        for col in num_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(self.df[col].mean())

        # 2. Categorical Imputation (Forward Fill with Backfill fallback)
        cat_cols = ['Stage_fear', 'Drained_after_socializing']
        for col in cat_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].ffill()
                if self.df[col].isnull().any():
                    self.df[col] = self.df[col].fillna(method='bfill')
                    # Ultimate fallback to mode if still NaN
                    if self.df[col].isnull().any():
                        self.df[col] = self.df[col].fillna(self.df[col].mode()[0])

        # 3. Manual Encoding (Creating _num columns)
        if 'Stage_fear' in self.df.columns:
            self.df['Stage_fear_num'] = self.df['Stage_fear'].map({'Yes': 1, 'No': 0})
            
        if 'Drained_after_socializing' in self.df.columns:
            self.df['Drained_after_socializing_num'] = self.df['Drained_after_socializing'].map({'Yes': 1, 'No': 0})

        # 4. Target Encoding (Extrovert=0, Introvert=1)
        if 'Personality' in self.df.columns:
            self.df['Personality_num'] = self.df['Personality'].map({'Extrovert': 0, 'Introvert': 1})

        # 5. Derived Features (Domain Knowledge)
        # Social Interaction Score: Aggregate of social activities
        self.df['Social_Interaction_Score'] = (
            self.df['Social_event_attendance'] + 
            self.df['Friends_circle_size'] + 
            self.df['Post_frequency']
        )

        # Online/Offline Ratio: Digital vs Physical social presence
        # Adding 1.0 to denominator to avoid division by zero
        self.df['Online_Offline_Ratio'] = (
            self.df['Post_frequency'] / (self.df['Going_outside'] + 1.0)
        )

        # Final NaN cleanup for derived columns
        derived_cols = ['Social_Interaction_Score', 'Online_Offline_Ratio', 
                        'Stage_fear_num', 'Drained_after_socializing_num']
        for col in derived_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(0)

        return self.df