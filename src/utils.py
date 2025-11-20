import json
import os

METRICS_FILE = "reports/metrics.json"

def save_metrics(model_name, metrics_dict):
    """
    Saves or updates model metrics in a JSON file for the dashboard.
    """
    os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)
    
    data = {}
    if os.path.exists(METRICS_FILE):
        try:
            with open(METRICS_FILE, 'r') as f:
                data = json.load(f)
        except:
            pass # Start fresh if corrupt
            
    data[model_name] = metrics_dict
    
    with open(METRICS_FILE, 'w') as f:
        json.dump(data, f, indent=4)

def load_all_metrics():
    """Loads all metrics for dashboard display."""
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, 'r') as f:
            return json.load(f)
    return {}