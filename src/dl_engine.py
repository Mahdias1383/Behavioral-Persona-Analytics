import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import plotly.express as px 
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class DeepLearningEngine:
    """
    Deep Learning Engine implementing the ANN architecture EXACTLY as per the reference notebook CELL.
    
    KEY CHANGE: Performs its own independent Preprocessing (StandardScaler & Split) inside,
    just like the reference notebook cell.
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
        self.scaler = StandardScaler() # ANN uses StandardScaler specifically in the ref cell

    def _build_model(self, input_dim):
        """
        Builds the sequential model exactly as described in the reference code.
        """
        model = Sequential()
        model.add(Dense(16, input_dim=input_dim, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def execute_ann_pipeline(self, df, target_col):
        """
        Runs the FULL ANN pipeline (Data Prep -> Split -> Build -> Train -> Eval)
        exactly as it appears in the standalone cell of the reference notebook.
        """
        print("\n🧠 Starting Independent ANN Pipeline (Reference Cell Logic)...")
        
        # --- Step 1: Define Features and Label (Local Definition) ---
        # The reference cell drops 'Personality_num' to form X, and uses it for y.
        # We assume df coming here has 'Personality_num' from Feature Engineering step.
        
        if f"{target_col}_num" not in df.columns:
             raise ValueError(f"Target column '{target_col}_num' missing.")
             
        y = df[f"{target_col}_num"]
        
        # Drop target related columns to form X (Local clean slate)
        # We remove ANY target related info to be safe and match 'X = df.drop(...)' logic
        cols_to_drop = [f"{target_col}_num", target_col]
        # Also remove any One-Hot encoded target columns if they exist from previous steps
        cols_to_drop.extend([c for c in df.columns if c.startswith(f"{target_col}_")])
        
        X = df.drop(columns=[c for c in cols_to_drop if c in df.columns], axis=1)
        
        # --- Step 2: Normalize features for ANN (StandardScaler) ---
        # Reference: scaler = StandardScaler(); X_scaled = scaler.fit_transform(X)
        print("   ℹ️ Scaling X with StandardScaler (Local)...")
        X_scaled = self.scaler.fit_transform(X)
        
        # --- Step 3: Split Data ---
        # Reference: X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        print("   ℹ️ Splitting Data (test_size=0.2, random_state=42)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # --- Step 4: Build Model ---
        self.model = self._build_model(input_dim=X_train.shape[1])
        
        # --- Step 5: Train ---
        # Reference: history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.1, verbose=1)
        print("   🧠 Training Model...")
        history = self.model.fit(
            X_train, y_train, 
            epochs=50, 
            batch_size=16, 
            validation_split=0.1, 
            verbose=1
        )
        
        self._save_model("personality_ann_reference_standalone.keras")
        self._plot_training_history(history)
        
        # --- Step 6: Evaluate ---
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
        fig.write_html(os.path.join(self.eval_dir, "ann_history_standalone.html"))

    def evaluate(self, X_test, y_test):
        print("\n📊 Evaluating Independent ANN Model...")
        y_pred_prob = self.model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype("int32").flatten()
        
        acc = accuracy_score(y_test, y_pred)
        print(f"🔹 ANN Accuracy: {acc}")
        print(classification_report(y_test, y_pred, target_names=['Extrovert', 'Introvert']))
        
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(cm, text_auto=True, title="Confusion Matrix - ANN (Standalone)", color_continuous_scale='Purples')
        fig.write_html(os.path.join(self.eval_dir, "cm_ANN_standalone.html"))