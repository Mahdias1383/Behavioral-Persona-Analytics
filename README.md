🧠 Introvert vs. Extrovert Prediction Pipeline
==============================================

📖 Overview
-----------

This repository contains a robust, modular Machine Learning pipeline designed to classify individuals as **Introverts** or **Extroverts**. Unlike simple analysis scripts, this project is engineered with scalability and reproducibility in mind, following standard software engineering practices for data science.

The core logic is built upon behavioral data features such as:

*   🕰️ **Time spent alone**
    
*   🎤 **Stage fear**
    
*   🔋 **Energy levels after socializing**
    
*   📱 **Social media post frequency**
    

🗂️ Project Structure
---------------------

The project follows a modular architecture to ensure separation of concerns:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   ├── src/  │   ├── data_loader.py    # Data ingestion logic  │   ├── preprocessing.py  # Feature engineering & encoding  │   └── model.py          # Random Forest implementation  ├── main.py               # Orchestration script (Entry Point)  ├── requirements.txt      # Dependency management  └── personality_dataset.csv   `

🚀 Quick Start
--------------

### 1\. Installation

Clone the repo and install dependencies:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   git clone [https://github.com/YOUR_USERNAME/Introvert-vs-Extrovert-Analysis.git](https://github.com/YOUR_USERNAME/Introvert-vs-Extrovert-Analysis.git)  cd Introvert-vs-Extrovert-Analysis  pip install -r requirements.txt   `

### 2\. Setup Data

Ensure your personality\_dataset.csv is placed in the root directory (or update the path in main.py).

### 3\. Run the Pipeline

Execute the main script to trigger loading, processing, training, and evaluation:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python main.py   `

📊 Model Performance
--------------------

The Random Forest Classifier was selected for its ability to handle non-linear relationships in behavioral data.

*   **Accuracy:** ~95%+ (on test set)
    
*   **Key Insight:** The feature Time\_spent\_Alone proved to be the strongest predictor of Introversion.
    

🛠 Technologies Used
--------------------

*   **Pandas & NumPy:** Data Manipulation
    
*   **Scikit-Learn:** Machine Learning
    
*   **Joblib:** Model Serialization
    

_Maintained by \[Your Name\] - Open for collaboration!_