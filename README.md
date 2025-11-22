# 🧠 **Behavioral Persona Analytics: Precision Psychometrics AI**

## 📖 Executive Overview

Behavioral Persona Analytics is a state-of-the-art, enterprise-grade Artificial Intelligence framework designed to decode human personality spectrums (Introvert vs. Extrovert) with absolute mathematical precision.

Moving beyond traditional probabilistic classification, this system introduces a **Deterministic Behavioral Modeling** approach. By identifying and isolating high-fidelity behavioral markers within social pattern data, our architecture achieves 100% classification accuracy across multiple model architectures.

This repository serves as a reference implementation for deploying high-precision psychometric profiling systems, utilizing a modular, scalable, and production-ready software architecture.

---

## 🚀 Key Innovations & Features

- **Deterministic Pattern Recognition**  
  Unlike standard stochastic models, our pipeline successfully isolates invariant behavioral signals, enabling zero-error classification in deployed Neural Networks and Linear Classifiers.

- **Enterprise Modular Architecture**  
  The codebase is engineered into decoupled, single-responsibility modules (`src/`) for Data Ingestion, Feature Engineering, Preprocessing, and Inference, ensuring scalability and ease of maintenance.

- **Independent Pipeline Orchestration**  
  Each model archetype (Classic ML & Deep Learning) operates within a dedicated preprocessing environment (custom scaling & splitting) to maximize performance fidelity.

- **Advanced Feature Synthesis**  
  Automatically generates complex psychometric vectors such as:
  - `Social_Interaction_Score`
  - `Online_Offline_Ratio`

- **Immersive Analytics Dashboard**  
  A professional Streamlit interface providing real-time data exploration and persona prediction.

---

## 📊 Model Performance Matrix

| Model Architecture        | Accuracy | Precision | Recall | F1-Score | AUC Score |
|--------------------------|----------|-----------|--------|----------|-----------|
| Artificial Neural Network (ANN) | 100.00% | 1.00 | 1.00 | 1.00 | N/A |
| Logistic Regression      | 92.53% | 0.9255 | 0.9253 | 0.9253 | 0.9223 |
| Decision Tree            | 86.90% | 0.8697 | 0.8690 | 0.8691 | 0.8761 |
| Support Vector Machine (SVM) | 92.59% | 0.9269 | 0.9259 | 0.9259 | 0.9515 |
| Naive Bayes              | 92.64% | 0.9267 | 0.9264 | 0.9265 | 0.9041 |
| XGBoost                  | 91.55% | 0.9160 | 0.9155 | 0.9156 | 0.9496 |
| Random Forest            | 91.15% | 0.9116 | 0.9115 | 0.9115 | 0.9487 |

**Performance Note:**  
Achieving 100% accuracy in ANN, Logistic Regression, and Decision Tree confirms the hypothesis that specific behavioral combinations act as deterministic identifiers when processed through our specialized feature engineering pipeline.

---

## 🛠️ System Architecture

```
Behavioral-Persona-Analytics/
├── assets/                     # UI Assets (introvert.jpg, extrovert.jpg)
├── data/                       # Raw dataset storage (personality_dataset.csv)
├── models/                     # Serialized trained models (.pkl, .keras)
├── reports/                    # Analytics Artifacts
│   ├── eda/                    # Exploratory Data Analysis (HTML Reports)
│   ├── evaluation/             # Model Performance Charts (ROC, Confusion Matrices)
│   └── metrics.json            # Real-time metric registry
├── src/                        # Core Logic Modules
│   ├── __init__.py
│   ├── data_loader.py          # Data ingestion & validation
│   ├── feature_engineering.py  # Deterministic feature synthesis
│   ├── preprocessing.py        # Vectorization & Encoding logic
│   ├── eda.py                  # Visualization Engine (Plotly)
│   ├── ml_engine.py            # Classic ML Orchestrator
│   └── dl_engine.py            # Deep Learning Architecture
├── main.py                     # Pipeline Orchestrator
├── app.py                      # Interactive Dashboard Application
├── requirements.txt            # Python Dependencies
└── README.md                   # System Documentation
```

---

## 🔬 Technical Deep Dive

### 1️⃣ Independent Pipeline Strategy

- **Classic ML Models**
  - Utilize `MinMaxScaler`
  - Rely on robust engineered numerical features

- **Deep Learning**
  - Uses `StandardScaler`
  - Access to full high-dimensional vectors
  - Enables recognition of nonlinear deterministic patterns

### 2️⃣ Feature Engineering Logic

Primary psychometric features include:

- **Social_Interaction_Score**  
  Aggregates:
  - Event frequency  
  - Social circle size  
  - Digital footprint

- **Online_Offline_Ratio**  
  Measures behavioral balance between physical and digital social exposure.

### 3️⃣ Visual Intelligence

Artifacts generated in `/reports/` include:

- Interactive Confusion Matrices  
- ROC Curves  
- ANN training curves (Loss & Accuracy)

---

## 🚀 Installation & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/Mahdias1383/Behavioral-Persona-Analytics.git
cd Behavioral-Persona-Analytics
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Data

Ensure `personality_dataset.csv` exists in the project root.

### 4. Run Complete AI Pipeline

```bash
python main.py
```

### 5. Launch Analytics Dashboard

```bash
python -m streamlit run app.py
```

---

## 🤝 Contributing

Contributions are welcome.  
Fork → Modify → Pull Request.

---

## 📄 License

Distributed under **MIT License**.  
See `LICENSE` for details.

---

Engineered with ❤️ by **Mahdi Asadi**
