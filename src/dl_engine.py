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
    
    REPLICATION NOTE: 
    This module explicitly replicates the data state of the reference study, 
    including preserving One-Hot encoded target columns if present.
    This is done to demonstrate how the reference 100% accuracy was achieved 
    (likely via data leakage), for educational and reproduction purposes.
    """
    def __init__(self, models_dir: str = "models", base_report_dir: str = "reports"):
        # Set seeds for reproducibility (Deterministic behavior)
        tf.random.set_seed(42)
        np.random.seed(42)
        
        self.models_dir = models_dir
        self.eval_dir = os.path.join(base_report_dir, "evaluation")
        
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.eval_dir, exist_ok=True)
        self.model = None
        self.scaler = StandardScaler()

    def _build_model(self, input_dim: int) -> Sequential:
        model = Sequential()
        model.add(Dense(16, input_dim=input_dim, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def execute_ann_pipeline(self, df: pd.DataFrame, target_col: str):
        print("\n🧠 Starting Independent ANN Pipeline (Exact State Replication)...")
        
        # 1. Define Label
        y_col = f"{target_col}_num"
        if y_col not in df.columns:
             le = LabelEncoder()
             y = le.fit_transform(df[target_col])
        else:
             y = df[y_col]
        
        # 2. Define Features (Replication Logic)
        # We verify if leakage columns exist to confirm reproduction state
        leakage_cols = [c for c in df.columns if c.startswith(f"{target_col}_") and c != y_col]
        if leakage_cols:
            print(f"   ⚠️ Leakage Replication: Including target-derived columns: {leakage_cols}")
        
        # Drop ONLY the numeric target label and original string target
        cols_to_drop = [y_col, target_col]
        existing_drop = [c for c in cols_to_drop if c in df.columns]
        X = df.drop(columns=existing_drop, axis=1)
        
        # Ensure numeric consistency
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        print(f"   ℹ️ ANN Features ({len(X.columns)})")
        
        # 3. Scale (StandardScaler)
        print("   ℹ️ Scaling...")
        X_scaled = self.scaler.fit_transform(X)
        
        # 4. Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # 5. Build & Train
        self.model = self._build_model(input_dim=X_train.shape[1])
        print("   🧠 Training...")
        history = self.model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.1, verbose=1)
        
        self._save_model("personality_ann_exact_replication.keras")
        self._plot_training_history(history)
        self.evaluate(X_test, y_test)

    def _save_model(self, filename):
        self.model.save(os.path.join(self.models_dir, filename))

    def _plot_training_history(self, history):
        hist = pd.DataFrame(history.history)
        hist['epoch'] = hist.index + 1
        
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Accuracy", "Loss"))
        fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['accuracy'], name='Train Acc'), row=1, col=1)
        fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['val_accuracy'], name='Val Acc'), row=1, col=1)
        fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['loss'], name='Train Loss'), row=1, col=2)
        fig.update_layout(title_text="ANN Training History", height=500)
        fig.write_html(os.path.join(self.eval_dir, "ann_history_interactive.html"))

    def evaluate(self, X_test, y_test):
        y_pred = (self.model.predict(X_test) > 0.5).astype("int32").flatten()
        acc = accuracy_score(y_test, y_pred)
        print(f"🔹 ANN Accuracy: {acc}")
        print(classification_report(y_test, y_pred))
        
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(cm, text_auto=True, title="Confusion Matrix - ANN", color_continuous_scale='Purples')
        fig.write_html(os.path.join(self.eval_dir, "cm_ANN_replication.html"))