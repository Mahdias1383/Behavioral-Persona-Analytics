import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import plotly.express as px 
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class DeepLearningEngine:
    """
    Deep Learning Engine implementing the ANN architecture.
    REPLICATION STRATEGY: Replicate the exact data state of the reference notebook cell,
    including potential data leakage (Target One-Hot columns) if they exist in the dataframe.
    """
    def __init__(self, models_dir="models", base_report_dir="reports"):
        # Set seeds for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)
        
        self.models_dir = models_dir
        self.eval_dir = os.path.join(base_report_dir, "evaluation")
        
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.eval_dir, exist_ok=True)
        self.model = None
        self.scaler = StandardScaler()

    def _build_model(self, input_dim):
        model = Sequential()
        model.add(Dense(16, input_dim=input_dim, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def execute_ann_pipeline(self, df, target_col):
        print("\n🧠 Starting Independent ANN Pipeline (Exact State Replication)...")
        
        # --- Step 1: Define Label ---
        # We need Personality_num as target
        if f"{target_col}_num" not in df.columns:
             le = LabelEncoder()
             y = le.fit_transform(df[target_col])
             # Add it back to df temporarily to match drop logic
             df[f"{target_col}_num"] = y
        else:
             y = df[f"{target_col}_num"]
        
        # --- Step 2: Define Features (X) ---
        # Reference logic: X = df.drop('Personality_num', axis=1)
        # This keeps EVERYTHING else.
        # If Preprocessing created 'Personality_Extrovert', it stays in X.
        # This creates Target Leakage, which explains the 100% accuracy.
        
        # Check if target one-hots exist
        target_one_hots = [c for c in df.columns if c.startswith(f"{target_col}_") and c != f"{target_col}_num"]
        if target_one_hots:
            print(f"   ⚠️ Potential Target Leakage Detected! Features include: {target_one_hots}")
        
        # Drop ONLY Personality_num (and the original string column if present)
        cols_to_drop = [f"{target_col}_num", target_col]
        existing_drop = [c for c in cols_to_drop if c in df.columns]
        
        X = df.drop(columns=existing_drop, axis=1)
        
        # Ensure numeric (One-Hot cols are bool/uint8, convert to float/int)
        X = X.apply(pd.to_numeric, errors='coerce')
        X = X.fillna(0) # Handle any conversion artifacts
        
        print(f"   ℹ️ ANN Input Features ({len(X.columns)}): {list(X.columns)}")
        
        # --- Step 3: Normalize features (StandardScaler) ---
        print("   ℹ️ Scaling X with StandardScaler...")
        X_scaled = self.scaler.fit_transform(X)
        
        # --- Step 4: Split Data ---
        print("   ℹ️ Splitting Data (test_size=0.2, random_state=42)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # --- Step 5: Build Model ---
        self.model = self._build_model(input_dim=X_train.shape[1])
        
        # --- Step 6: Train ---
        print("   🧠 Training Model...")
        history = self.model.fit(
            X_train, y_train, 
            epochs=50, 
            batch_size=16, 
            validation_split=0.1, 
            verbose=1
        )
        
        self._save_model("personality_ann_exact_replication.keras")
        self._plot_training_history(history)
        
        # --- Step 7: Evaluate ---
        self.evaluate(X_test, y_test)

    def _save_model(self, filename):
        path = os.path.join(self.models_dir, filename)
        self.model.save(path)
        print(f"💾 ANN Model saved to: {path}")

    def _plot_training_history(self, history):
        hist_df = pd.DataFrame(history.history)
        hist_df['epoch'] = hist_df.index + 1
        
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Model Accuracy", "Model Loss"))
        fig.add_trace(go.Scatter(x=hist_df['epoch'], y=hist_df['accuracy'], name='Train Acc'), row=1, col=1)
        fig.add_trace(go.Scatter(x=hist_df['epoch'], y=hist_df['val_accuracy'], name='Val Acc'), row=1, col=1)
        fig.add_trace(go.Scatter(x=hist_df['epoch'], y=hist_df['loss'], name='Train Loss'), row=1, col=2)
        fig.add_trace(go.Scatter(x=hist_df['epoch'], y=hist_df['val_loss'], name='Val Loss'), row=1, col=2)
        fig.update_layout(title_text="ANN Training History", height=500)
        fig.write_html(os.path.join(self.eval_dir, "ann_history_interactive.html"))

    def evaluate(self, X_test, y_test):
        print("\n📊 Evaluating Independent ANN Model...")
        y_pred_prob = self.model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype("int32").flatten()
        
        acc = accuracy_score(y_test, y_pred)
        print(f"🔹 ANN Accuracy: {acc}")
        print(classification_report(y_test, y_pred, target_names=['Extrovert', 'Introvert']))
        
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(cm, text_auto=True, title="Confusion Matrix - ANN (Replication)", color_continuous_scale='Purples')
        fig.write_html(os.path.join(self.eval_dir, "cm_ANN_replication.html"))