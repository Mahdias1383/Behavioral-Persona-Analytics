import os
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Tuple
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_curve, auc)

class MLEngine:
    """
    Manages training and evaluation of Classic ML models.
    Implements Independent Pipelines (Local Scaling/Splitting) for each model.
    """
    def __init__(self, models_dir: str = "models", base_report_dir: str = "reports"):
        self.models_dir = models_dir
        self.eval_dir = os.path.join(base_report_dir, "evaluation")
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.eval_dir, exist_ok=True)
        
        self.metrics_summary = {
            'Model': [], 'Accuracy': [], 'Precision': [], 
            'Recall': [], 'F1 Score': [], 'AUC': []
        }

    def _prepare_data_locally(self, df: pd.DataFrame, target_col: str, 
                              test_size: float, random_state: int, 
                              scaler_type: str = 'minmax') -> Tuple:
        """
        Prepares a local subset of data for a specific model.
        """
        # 1. Target
        y_col = f"{target_col}_num"
        # Fallback if FeatureEng didn't create it (rare)
        if y_col not in df.columns:
             y = df[target_col].astype('category').cat.codes
        else:
             y = df[y_col]
        
        # 2. Features
        # Exclude target and derived target columns to avoid leakage in classic models
        cols_to_drop = [y_col, target_col]
        cols_to_drop.extend([c for c in df.columns if c.startswith(f"{target_col}_")])
        
        X = df.drop(columns=[c for c in cols_to_drop if c in df.columns], axis=1)
        # Ensure numeric
        X = X.select_dtypes(include=['number'])
        
        # 3. Scaling
        if scaler_type == 'minmax':
             scaler = MinMaxScaler()
             X = scaler.fit_transform(X)
        elif scaler_type == 'standard':
             scaler = StandardScaler()
             X = scaler.fit_transform(X)
             
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def run_all_classic_models(self, df: pd.DataFrame, target_col: str):
        """
        Orchestrates the training of all classic models.
        """
        print("\n🚀 Running Classic Models (Full Features, Metrics & ROC Curves)...")
        target_names = ['Extrovert', 'Introvert']

        # Configuration: (ModelClass, Name, TestSize, RandomState, Kwargs)
        configs = [
            (LogisticRegression, "Logistic_Regression", 0.3, 42, {'max_iter': 1000}),
            (SVC, "SVM", 0.2, 7, {'kernel': 'rbf', 'probability': True}),
            (DecisionTreeClassifier, "Decision_Tree", 0.2, 7, {}),
            (RandomForestClassifier, "Random_Forest", 0.3, 42, {'n_estimators': 100}),
            (GaussianNB, "Naive_Bayes", 0.3, 42, {}),
            (XGBClassifier, "XGBoost", 0.2, 42, {'eval_metric': 'logloss'})
        ]

        for model_cls, name, test_size, r_state, kwargs in configs:
            # Prepare data locally
            X_train, X_test, y_train, y_test = self._prepare_data_locally(
                df, target_col, test_size, r_state, 'minmax'
            )
            
            # Instantiate (Handle NaiveBayes no random_state)
            if name == "Naive_Bayes":
                model = model_cls(**kwargs)
            else:
                model = model_cls(random_state=r_state, **kwargs)
            
            self._train_eval(model, name, X_train, X_test, y_train, y_test, target_names)

        self._plot_comparison()

    def _train_eval(self, model, name, X_train, X_test, y_train, y_test, target_names):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # AUC
        roc_auc = 0.5
        y_score = None
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
            
        if y_score is not None:
            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_auc = auc(fpr, tpr)
            self._plot_roc_curve(fpr, tpr, roc_auc, name)

        # Summary
        pretty_name = name.replace('_', ' ')
        self.metrics_summary['Model'].append(pretty_name)
        self.metrics_summary['Accuracy'].append(acc)
        self.metrics_summary['Precision'].append(prec)
        self.metrics_summary['Recall'].append(rec)
        self.metrics_summary['F1 Score'].append(f1)
        self.metrics_summary['AUC'].append(roc_auc)
        
        print(f"\n📊 {pretty_name} Results:")
        print(f"   Accuracy:  {acc:.4f}")
        print(f"   Precision: {prec:.4f}")
        print(f"   Recall:    {rec:.4f}")
        print(f"   F1 Score:  {f1:.4f}")
        print(f"   AUC Score: {roc_auc:.4f}")
        
        # Save artifacts
        joblib.dump(model, os.path.join(self.models_dir, f"{name}.pkl"))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(cm, text_auto=True, x=target_names, y=target_names, 
                        title=f"Confusion Matrix - {pretty_name}", color_continuous_scale='Blues')
        fig.write_html(os.path.join(self.eval_dir, f"cm_{name}.html"))

    def _plot_roc_curve(self, fpr, tpr, roc_auc, name):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'AUC={roc_auc:.2f}', line=dict(color='darkorange')))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(color='navy', dash='dash')))
        fig.update_layout(title=f"ROC Curve - {name}", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", width=700, height=500)
        fig.write_html(os.path.join(self.eval_dir, f"roc_curve_{name}.html"))

    def _plot_comparison(self):
        df = pd.DataFrame(self.metrics_summary)
        df_melted = df.melt(id_vars="Model", var_name="Metric", value_name="Score")
        fig = px.bar(df_melted, x="Model", y="Score", color="Metric", barmode="group", title="Model Comparison")
        fig.update_yaxes(range=[0.8, 1.05])
        fig.write_html(os.path.join(self.eval_dir, "model_comparison_full.html"))