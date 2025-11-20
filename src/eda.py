import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

class EDAReport:
    """
    Advanced EDA module using Plotly.
    Handles Data Inspection, Anomaly Detection, and Correlation Analysis.
    Saves interactive HTML reports.
    """
    def __init__(self, df: pd.DataFrame, base_report_dir="reports"):
        self.df = df
        self.eda_dir = os.path.join(base_report_dir, "eda")
        os.makedirs(self.eda_dir, exist_ok=True)

    def generate_inspection_report(self):
        """
        Generates a text report summarizing dataset structure.
        """
        print("   📊 Generating Data Inspection Report...")
        report_path = os.path.join(self.eda_dir, "data_inspection.txt")
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=== DATA INSPECTION REPORT ===\n\n")
            f.write(f"Shape: {self.df.shape}\n\n")
            f.write(f"Columns: {list(self.df.columns)}\n\n")
            f.write("--- Missing Values ---\n")
            f.write(f"{self.df.isnull().sum()}\n\n")
            f.write("--- Descriptive Statistics ---\n")
            f.write(f"{self.df.describe()}\n")
            
        print(f"   ✅ Inspection report saved to {report_path}")

    def plot_anomalies_boxplot(self):
        """
        Generates interactive boxplots for numerical columns.
        """
        print("   📊 Generating Anomaly Detection Boxplots...")
        numeric_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                        'Friends_circle_size', 'Post_frequency', 'Social_Interaction_Score', 'Online_Offline_Ratio']
        
        # Filter columns that actually exist
        numeric_cols = [c for c in numeric_cols if c in self.df.columns]

        for col in numeric_cols:
            fig = px.box(self.df, y=col, title=f"Anomalies in {col}", points="all")
            fig.write_html(os.path.join(self.eda_dir, f"boxplot_anomaly_{col}.html"))

    def plot_categorical_anomalies(self):
        """
        Generates bar charts for categorical columns.
        """
        print("   📊 Generating Categorical Distribution Plots...")
        cat_cols = ['Stage_fear', 'Drained_after_socializing', 'Personality']
        
        cat_cols = [c for c in cat_cols if c in self.df.columns]

        for col in cat_cols:
            counts = self.df[col].value_counts(dropna=False).reset_index()
            counts.columns = [col, 'Count']
            counts[col] = counts[col].fillna('Missing')
            
            fig = px.bar(counts, x=col, y='Count', 
                         title=f"Distribution of {col}",
                         text='Count', color='Count')
            fig.write_html(os.path.join(self.eda_dir, f"barplot_counts_{col}.html"))

    def plot_correlation_matrix(self, title_suffix=""):
        """
        Generates a correlation heatmap for numerical features.
        """
        print(f"   📊 Generating Correlation Matrix {title_suffix}...")
        
        numeric_df = self.df.select_dtypes(include=['number'])
        
        if numeric_df.empty:
            return

        corr = numeric_df.corr()
        
        fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                        title=f"Correlation Matrix {title_suffix}",
                        color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
        
        filename = f"correlation_matrix{title_suffix.replace(' ', '_').lower()}.html"
        fig.write_html(os.path.join(self.eda_dir, filename))