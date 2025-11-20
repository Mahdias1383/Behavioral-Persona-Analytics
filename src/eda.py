import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

class EDAReport:
    """
    Advanced EDA module using Plotly.
    Handles Data Inspection, Anomaly Detection, and Correlation Analysis.
    Saves interactive HTML reports in reports/eda/.
    """
    def __init__(self, df: pd.DataFrame, base_report_dir="reports"):
        self.df = df
        self.eda_dir = os.path.join(base_report_dir, "eda")
        # Ensure directory exists
        os.makedirs(self.eda_dir, exist_ok=True)

    def generate_inspection_report(self):
        """
        Generates a text report summarizing dataset structure.
        """
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
            print(f"   ✅ Inspection report saved to {report_path}")
        except Exception as e:
            print(f"   ❌ Error saving inspection report: {e}")

    def plot_anomalies_boxplot(self):
        """
        Generates interactive boxplots for numerical columns.
        """
        print("   📊 Generating Anomaly Detection Boxplots...")
        # Select strictly numeric columns
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_cols:
            print("   ⚠️ No numeric columns found for boxplots.")
            return

        for col in numeric_cols:
            try:
                fig = px.box(self.df, y=col, title=f"Anomalies in {col}", points="all")
                fig.write_html(os.path.join(self.eda_dir, f"boxplot_anomaly_{col}.html"))
            except Exception as e:
                print(f"   ⚠️ Could not plot boxplot for {col}: {e}")

    def plot_categorical_anomalies(self):
        """
        Generates bar charts for categorical columns.
        """
        print("   📊 Generating Categorical Distribution Plots...")
        # Select categorical/object columns + specific known categoricals
        # Logic: Object columns OR columns with few unique values (like encoded targets)
        
        # Explicit list based on dataset knowledge
        potential_cats = ['Stage_fear', 'Drained_after_socializing', 'Personality', 
                          'Stage_fear_num', 'Drained_after_socializing_num', 'Personality_num']
        
        cat_cols = [c for c in potential_cats if c in self.df.columns]
        
        # Add object columns if not in list
        object_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        for c in object_cols:
            if c not in cat_cols:
                cat_cols.append(c)

        for col in cat_cols:
            try:
                counts = self.df[col].value_counts(dropna=False).reset_index()
                counts.columns = [col, 'Count']
                counts[col] = counts[col].fillna('Missing')
                
                fig = px.bar(counts, x=col, y='Count', 
                             title=f"Distribution of {col}",
                             text='Count', color='Count')
                fig.write_html(os.path.join(self.eda_dir, f"barplot_counts_{col}.html"))
            except Exception as e:
                print(f"   ⚠️ Could not plot barplot for {col}: {e}")

    def plot_correlation_matrix(self, title_suffix=""):
        """
        Generates a correlation heatmap for numerical features.
        """
        print(f"   📊 Generating Correlation Matrix {title_suffix}...")
        
        numeric_df = self.df.select_dtypes(include=['number'])
        
        if numeric_df.empty:
            print("   ⚠️ No numerical columns for correlation.")
            return

        try:
            corr = numeric_df.corr()
            
            fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                            title=f"Correlation Matrix {title_suffix}",
                            color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
            
            # Sanitize filename
            safe_suffix = title_suffix.replace(' ', '_').replace('(', '').replace(')', '').lower()
            filename = f"correlation_matrix{safe_suffix}.html"
            fig.write_html(os.path.join(self.eda_dir, filename))
        except Exception as e:
            print(f"   ❌ Error generating correlation matrix: {e}")