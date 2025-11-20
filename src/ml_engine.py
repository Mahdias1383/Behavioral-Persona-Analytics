import os
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc

class MLEngine:
    """
    ML Engine with independent pipelines.
    UPDATED: Prints ALL metrics to console (Accuracy, Precision, Recall, F1, AUC).
    """
    def __init__(self, models_dir="models", base_report_dir="reports"):
        self.models_dir = models_dir
        self.eval_dir = os.path.join(base_report_dir, "evaluation")
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.eval_dir, exist_ok=True)
        # Initialize storage for all metrics
        self.metrics_summary = {
            'Model': [], 
            'Accuracy': [],
            'Precision': [],
            'Recall': [],
            'F1 Score': [],
            'AUC': [] 
        }

    def _prepare_data_locally(self, df, target_col, test_size, random_state, scaler_type='minmax'):
        """
        Prepares data locally using strictly numeric features.
        """
        # 1. Define Target
        if f"{target_col}_num" not in df.columns:
             from sklearn.preprocessing import LabelEncoder
             le = LabelEncoder()
             y = le.fit_transform(df[target_col])
        else:
             y = df[f"{target_col}_num"]
        
        # 2. Define Features (X)
        cols_to_drop = [f"{target_col}_num", target_col]
        cols_to_drop.extend([c for c in df.columns if c.startswith(f"{target_col}_")])
        
        X = df.drop(columns=[c for c in cols_to_drop if c in df.columns], axis=1)
        
        # FORCE selection of numeric types only
        X = X.select_dtypes(include=['number'])
        
        # 3. Scale
        if scaler_type == 'minmax':
             scaler = MinMaxScaler()
             X = scaler.fit_transform(X)
        elif scaler_type == 'standard':
             scaler = StandardScaler()
             X = scaler.fit_transform(X)
             
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def run_all_classic_models(self, df, target_col):
        print("\n🚀 Running Classic Models (Full Features, Metrics & ROC Curves)...")
        target_names = ['Extrovert', 'Introvert']

        # Helper to run pipeline
        def run_model(model_class, name, test_size, random_state, **kwargs):
            X_train, X_test, y_train, y_test = self._prepare_data_locally(df, target_col, test_size, random_state, 'minmax')
            
            if name == "Naive_Bayes":
                 model = model_class(**kwargs)
            else:
                 model = model_class(random_state=random_state, **kwargs)
                 
            self._train_eval(model, name, X_train, X_test, y_train, y_test, target_names)

        # --- 1. Logistic Regression ---
        run_model(LogisticRegression, "Logistic_Regression", 0.3, 42, max_iter=1000)

        # --- 2. SVM ---
        run_model(SVC, "SVM", 0.2, 7, kernel='rbf', probability=True)

        # --- 3. Decision Tree ---
        run_model(DecisionTreeClassifier, "Decision_Tree", 0.2, 7)

        # --- 4. Random Forest ---
        run_model(RandomForestClassifier, "Random_Forest", 0.3, 42, n_estimators=100)

        # --- 5. Naive Bayes ---
        run_model(GaussianNB, "Naive_Bayes", 0.3, 42)
        
        # --- 6. XGBoost ---
        run_model(XGBClassifier, "XGBoost", 0.2, 42, eval_metric='logloss')

        self._plot_comparison()

    def _train_eval(self, model, name, X_train, X_test, y_train, y_test, target_names):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate probability scores for ROC
        y_score = None
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1] 
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
            
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Calculate AUC
        roc_auc = 0.5
        if y_score is not None:
            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_auc = auc(fpr, tpr)
            self._plot_roc_curve(fpr, tpr, roc_auc, name)
        
        # Store metrics
        self.metrics_summary['Model'].append(name.replace('_', ' '))
        self.metrics_summary['Accuracy'].append(acc)
        self.metrics_summary['Precision'].append(prec)
        self.metrics_summary['Recall'].append(rec)
        self.metrics_summary['F1 Score'].append(f1)
        self.metrics_summary['AUC'].append(roc_auc)
        
        # --- UPDATED PRINT BLOCK ---
        print(f"\n📊 {name.replace('_', ' ')} Results:")
        print(f"   Accuracy:  {acc:.4f}")
        print(f"   Precision: {prec:.4f}")
        print(f"   Recall:    {rec:.4f}")
        print(f"   F1 Score:  {f1:.4f}")
        print(f"   AUC Score: {roc_auc:.4f}")
        
        joblib.dump(model, os.path.join(self.models_dir, f"{name}.pkl"))
        
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(cm, text_auto=True, x=target_names, y=target_names, 
                        title=f"Confusion Matrix - {name}", color_continuous_scale='Blues')
        fig.write_html(os.path.join(self.eval_dir, f"cm_{name}.html"))

    def _plot_roc_curve(self, fpr, tpr, roc_auc, name):
        """
        Generates and saves an interactive ROC Curve plot.
        """
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC = {roc_auc:.2f})', line=dict(color='darkorange', width=2)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Guess', line=dict(color='navy', width=2, dash='dash')))
        fig.update_layout(title=f"ROC Curve - {name.replace('_', ' ')}", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", width=700, height=500)
        save_path = os.path.join(self.eval_dir, f"roc_curve_{name}.html")
        fig.write_html(save_path)

    def _plot_comparison(self):
        """
        Plots a grouped bar chart comparing ALL metrics for all models.
        """
        df_metrics = pd.DataFrame(self.metrics_summary)
        df_melted = df_metrics.melt(id_vars="Model", var_name="Metric", value_name="Score")
        fig = px.bar(df_melted, x="Model", y="Score", color="Metric", barmode="group", title="Comprehensive Model Comparison", text_auto='.2f', color_discrete_sequence=px.colors.qualitative.Safe)
        fig.update_yaxes(range=[0.8, 1.05])
        fig.write_html(os.path.join(self.eval_dir, "model_comparison_full.html"))