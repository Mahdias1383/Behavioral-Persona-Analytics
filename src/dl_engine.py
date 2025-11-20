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

class DeepLearningEngine:
    """
    Deep Learning Engine that replicates the reference notebook's code block EXACTLY.
    We literally copy-pasted the logic to ensure 100% reproducibility.
    """
    def __init__(self, models_dir="models", base_report_dir="reports"):
        # Set seeds just in case
        tf.random.set_seed(42)
        np.random.seed(42)
        
        self.models_dir = models_dir
        self.eval_dir = os.path.join(base_report_dir, "evaluation")
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.eval_dir, exist_ok=True)

    def execute_ann_pipeline(self, df, target_col):
        print("\n🧠 Starting ANN Pipeline (The Exact Copy Method)...")

        # --- EXACT COPY FROM REFERENCE SNIPPET STARTS HERE ---
        
        # Step 2: Load Data (Assuming df is already cleaned and processed)
        # NOTE: We need 'Personality_num' in df. 
        if 'Personality_num' not in df.columns:
             # If not present, create it exactly as reference does
             if target_col in df.columns:
                 df['Personality_num'] = df[target_col].map({'Extrovert': 0, 'Introvert': 1})
             else:
                 raise ValueError("Original target column missing to create Personality_num")

        # Step 3: Define Features and Label
        # X = df.drop('Personality_num', axis=1) 
        # CAREFUL: The reference snippet assumes df ONLY has features + Personality_num.
        # Our df might have extra columns like 'Personality', 'Stage_fear', etc.
        # We must clean df to match reference state: only numeric features + target num.
        
        # Let's construct the exact feature set expected:
        # "Time_spent_Alone", "Social_event_attendance", "Going_outside", "Friends_circle_size", "Post_frequency"
        # Plus "Stage_fear_num", "Drained_after_socializing_num" if they exist in ref X.
        # The snippet creates X by dropping 'Personality_num'. 
        # So we must drop everything else that is NOT a feature.
        
        cols_to_drop = ['Personality_num', target_col]
        # Also drop raw categorical columns if they still exist
        raw_cats = ['Stage_fear', 'Drained_after_socializing']
        cols_to_drop.extend([c for c in raw_cats if c in df.columns])
        
        X = df.drop(columns=[c for c in cols_to_drop if c in df.columns], axis=1)
        y = df['Personality_num']
        
        print(f"   ℹ️ Features used for ANN: {list(X.columns)}")

        # Step 4: Normalize features for ANN (best practice)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Step 5: Split Data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Step 6: Build ANN Model
        model = Sequential()
        model.add(Dense(16, input_dim=X.shape[1], activation='relu'))  # First hidden layer
        model.add(Dense(8, activation='relu'))                          # Second hidden layer
        model.add(Dense(1, activation='sigmoid'))                       # Output layer (binary classification)

        # Step 7: Compile the Model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Step 8: Train the Model
        print("   🧠 Training...")
        history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.1, verbose=1)

        # Step 9: Evaluate the ANN Model
        y_pred_ann = (model.predict(X_test) > 0.5).astype("int32")
        print("🔹 ANN Accuracy:", accuracy_score(y_test, y_pred_ann))
        print(classification_report(y_test, y_pred_ann, target_names=['Extrovert', 'Introvert']))
        
        # --- EXACT COPY ENDS HERE ---
        
        # Save artifacts (My addition)
        model.save(os.path.join(self.models_dir, "personality_ann_exact.keras"))
        
        cm = confusion_matrix(y_test, y_pred_ann)
        fig = px.imshow(cm, text_auto=True, title="Confusion Matrix - ANN (Exact Copy)", color_continuous_scale='Purples')
        fig.write_html(os.path.join(self.eval_dir, "cm_ANN_exact.html"))