import pandas as pd
import numpy as np

class FeatureEngineer:
    """
    Handles specific feature engineering steps found in the reference notebook.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def apply_engineering(self):
        print("   🛠️ Applying Feature Engineering...")
        
        # 1. Impute Numerical Columns (Mean)
        num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                    'Friends_circle_size', 'Post_frequency']
        for col in num_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(self.df[col].mean())

        # 2. Impute Categorical Columns (ffill)
        cat_cols = ['Stage_fear', 'Drained_after_socializing']
        for col in cat_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].ffill()
                # Fallback
                if self.df[col].isnull().any():
                    self.df[col] = self.df[col].fillna(method='bfill')
                    if self.df[col].isnull().any():
                        self.df[col] = self.df[col].fillna(self.df[col].mode()[0])

        # 3. Manual Mapping to create _num columns
        if 'Stage_fear' in self.df.columns:
            self.df['Stage_fear_num'] = self.df['Stage_fear'].map({'Yes': 1, 'No': 0})
            
        if 'Drained_after_socializing' in self.df.columns:
            self.df['Drained_after_socializing_num'] = self.df['Drained_after_socializing'].map({'Yes': 1, 'No': 0})

        # 4. Encode Target
        # CRITICAL: We strictly map Extrovert->0, Introvert->1
        if 'Personality' in self.df.columns:
            self.df['Personality_num'] = self.df['Personality'].map({'Extrovert': 0, 'Introvert': 1})

        # --- Derived Features ---
        self.df['Social_Interaction_Score'] = (
            self.df['Social_event_attendance'] + 
            self.df['Friends_circle_size'] + 
            self.df['Post_frequency']
        )

        # Avoid division by zero
        self.df['Online_Offline_Ratio'] = (
            self.df['Post_frequency'] / (self.df['Going_outside'] + 1.0)
        )

        # CRITICAL CHANGE: Do NOT fillna(0) blindly on everything!
        # Only fill NaNs in the new derived numeric columns if they exist
        new_cols = ['Social_Interaction_Score', 'Online_Offline_Ratio', 'Stage_fear_num', 'Drained_after_socializing_num']
        for col in new_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(0)

        return self.df