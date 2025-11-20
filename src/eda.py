import os
import pandas as pd
import plotly.express as px
import plotly.io as pio

# Set default template
pio.templates.default = "plotly_dark"

class EDAReport:
    """
    Exploratory Data Analysis module using Plotly.
    Generates interactive HTML reports with optimized layouts.
    """
    def __init__(self, df: pd.DataFrame, base_report_dir: str = "reports"):
        self.df = df
        self.eda_dir = os.path.abspath(os.path.join(os.getcwd(), base_report_dir, "eda"))
        if not os.path.exists(self.eda_dir):
            os.makedirs(self.eda_dir)

    def generate_inspection_report(self):
        """Generates a text summary of dataset statistics."""
        print("   📊 Generating Data Inspection Report...")
        report_path = os.path.join(self.eda_dir, "data_inspection.txt")
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write("=== DATA INSPECTION REPORT ===\n\n")
                f.write(f"Shape: {self.df.shape}\n\n")
                f.write(f"Columns: {list(self.df.columns)}\n\n")
                f.write("--- Missing Values ---\n")
                f.write(f"{self.df.isnull().sum()}\n\n")
                f.write("--- Descriptive Statistics ---\n")
                f.write(f"{self.df.describe()}\n")
            print(f"   ✅ Report saved: {report_path}")
        except Exception as e:
            print(f"   ⚠️ Error saving report: {e}")

    def plot_anomalies_boxplot(self):
        """Generates interactive boxplots for numerical columns."""
        print("   📊 Generating Anomaly Detection Boxplots...")
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        for col in numeric_cols:
            try:
                fig = px.box(
                    self.df, 
                    y=col, 
                    title=f"Anomaly Analysis: {col}",
                    points="outliers",
                    color_discrete_sequence=["#FF4B4B"]
                )
                # Fixed layout dimensions
                fig.update_layout(
                    autosize=False,
                    width=800,
                    height=500,
                    margin=dict(l=80, r=40, b=60, t=80),
                    yaxis_title=col.replace('_', ' ')
                )
                fig.write_html(os.path.join(self.eda_dir, f"boxplot_anomaly_{col}.html"))
            except Exception:
                continue

    def plot_categorical_anomalies(self):
        """Generates bar charts for categorical distributions."""
        print("   📊 Generating Categorical Distributions...")
        cat_cols = [c for c in self.df.columns if self.df[c].dtype == 'object']
        
        if 'Personality' not in cat_cols and 'Personality' in self.df.columns:
            cat_cols.append('Personality')

        for col in cat_cols:
            try:
                counts = self.df[col].value_counts(dropna=False).reset_index()
                counts.columns = [col, 'Count']
                counts[col] = counts[col].fillna('Missing')
                
                fig = px.bar(
                    counts, 
                    x=col, 
                    y='Count', 
                    title=f"Distribution: {col}",
                    text='Count',
                    color='Count',
                    color_continuous_scale='Viridis'
                )
                
                fig.update_layout(
                    autosize=False,
                    width=800,
                    height=500,
                    margin=dict(l=60, r=40, b=100, t=80),
                    xaxis_title=col.replace('_', ' '),
                    yaxis_title="Count",
                    showlegend=False
                )
                fig.write_html(os.path.join(self.eda_dir, f"barplot_counts_{col}.html"))
            except Exception:
                continue

    def plot_correlation_matrix(self, title_suffix: str = ""):
        """Generates a correlation heatmap."""
        print(f"   📊 Generating Correlation Matrix {title_suffix}...")
        numeric_df = self.df.select_dtypes(include=['number'])
        
        if numeric_df.empty:
            return

        try:
            corr = numeric_df.corr()
            
            fig = px.imshow(
                corr, 
                text_auto=".2f", 
                aspect="auto",
                title=f"Correlation Matrix {title_suffix}",
                color_continuous_scale='RdBu_r',
                zmin=-1, 
                zmax=1
            )
            
            fig.update_layout(
                autosize=False,
                width=950,
                height=800,
                margin=dict(l=150, r=50, b=150, t=100)
            )
            
            safe_suffix = title_suffix.replace(' ', '_').replace('(', '').replace(')', '').lower()
            fig.write_html(os.path.join(self.eda_dir, f"correlation_matrix{safe_suffix}.html"))
        except Exception:
            pass