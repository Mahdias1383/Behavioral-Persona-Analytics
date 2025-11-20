import os
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class MLEngine:
    """
    ML Engine where each model executes its own independent data preparation pipeline
    exactly as shown in separate cells of the reference notebook.
    """
    def __init__(self, models_dir="models", base_report_dir="reports"):
        self.models_dir = models_dir
        self.eval_dir = os.path.join(base_report_dir, "evaluation")
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.eval_dir, exist_ok=True)
        self.metrics_summary = {'Model': [], 'Accuracy': []}

    def _prepare_data_locally(self, df, target_col, test_size, random_state, scaler_type=None):
        """Helper to prep data locally for each model."""
        if f"{target_col}_num" not in df.columns:
             raise ValueError("Target num column missing.")
        y = df[f"{target_col}_num"]
        
        cols_to_drop = [f"{target_col}_num", target_col]
        cols_to_drop.extend([c for c in df.columns if c.startswith(f"{target_col}_")])
        X = df.drop(columns=[c for c in cols_to_drop if c in df.columns], axis=1)
        
        # Some models in reference might use scaling, some might not.
        # Based on snippets, ANN used StandardScaler explicitly.
        # Others used X directly from a previous global processing step (likely MinMaxScaler).
        # To be safe and robust, we can apply MinMaxScaler here if scaler_type is 'minmax'.
        
        if scaler_type == 'minmax':
             scaler = MinMaxScaler()
             X = scaler.fit_transform(X)
        elif scaler_type == 'standard':
             scaler = StandardScaler()
             X = scaler.fit_transform(X)
             
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def run_all_classic_models(self, df, target_col):
        print("\n🚀 Running Classic Models (Independent Pipelines)...")
        target_names = ['Extrovert', 'Introvert']

        # --- 1. Logistic Regression ---
        # Snippet: test_size=0.3, random_state=42
        X_train, X_test, y_train, y_test = self._prepare_data_locally(df, target_col, 0.3, 42)
        self._train_eval(LogisticRegression(max_iter=1000, random_state=42), "Logistic_Regression", X_train, X_test, y_train, y_test, target_names)

        # --- 2. SVM ---
        # Snippet: test_size=0.2, random_state=7
        X_train, X_test, y_train, y_test = self._prepare_data_locally(df, target_col, 0.2, 7)
        self._train_eval(SVC(kernel='rbf', random_state=7), "SVM", X_train, X_test, y_train, y_test, target_names)

        # --- 3. Decision Tree ---
        # Snippet: test_size=0.2, random_state=7
        X_train, X_test, y_train, y_test = self._prepare_data_locally(df, target_col, 0.2, 7)
        self._train_eval(DecisionTreeClassifier(random_state=7), "Decision_Tree", X_train, X_test, y_train, y_test, target_names)

        # --- 4. Random Forest ---
        # Snippet: test_size=0.3, random_state=42
        X_train, X_test, y_train, y_test = self._prepare_data_locally(df, target_col, 0.3, 42)
        self._train_eval(RandomForestClassifier(n_estimators=100, random_state=42), "Random_Forest", X_train, X_test, y_train, y_test, target_names)

        # --- 5. Naive Bayes ---
        # Snippet: test_size=0.3, random_state=42
        X_train, X_test, y_train, y_test = self._prepare_data_locally(df, target_col, 0.3, 42)
        self._train_eval(GaussianNB(), "Naive_Bayes", X_train, X_test, y_train, y_test, target_names)
        
        # --- 6. XGBoost (Bonus) ---
        X_train, X_test, y_train, y_test = self._prepare_data_locally(df, target_col, 0.2, 42)
        self._train_eval(XGBClassifier(eval_metric='logloss', random_state=42), "XGBoost", X_train, X_test, y_train, y_test, target_names)

        self._plot_comparison()

    def _train_eval(self, model, name, X_train, X_test, y_train, y_test, target_names):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        self.metrics_summary['Model'].append(name)
        self.metrics_summary['Accuracy'].append(acc)
        
        print(f"\n📊 {name.replace('_', ' ')} Accuracy: {acc:.2f}")
        # Save artifacts
        joblib.dump(model, os.path.join(self.models_dir, f"{name}.pkl"))
        
        # Plot CM
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(cm, text_auto=True, x=target_names, y=target_names, 
                        title=f"Confusion Matrix - {name}", color_continuous_scale='Blues')
        fig.write_html(os.path.join(self.eval_dir, f"cm_{name}.html"))

    def _plot_comparison(self):
        df_metrics = pd.DataFrame(self.metrics_summary)
        fig = px.bar(df_metrics, x="Model", y="Accuracy", title="Model Comparison (Independent Pipelines)", text_auto='.2f')
        fig.update_yaxes(range=[0.8, 1.05])
        fig.write_html(os.path.join(self.eval_dir, "model_comparison_independent.html"))