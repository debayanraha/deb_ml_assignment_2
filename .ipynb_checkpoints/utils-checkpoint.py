import streamlit as st
import subprocess
import joblib
import pandas as pd
from pathlib import Path
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert import HTMLExporter
from traitlets.config import Config
import os
import base64
from PIL import Image
import io

MODEL_DIR = Path("model")
DATA_DIR = Path("data")


def get_model_path(model_name):
    name_map = {
        "Logistic Regression": "logistic_regression.joblib",
        "Decision Tree": "decision_tree.joblib",
        "KNN": "knn.joblib",
        "Naive Bayes": "naive_bayes.joblib",
        "Random Forest": "random_forest.joblib",
        "XGBoost": "xgboost.joblib"
    }
    return os.path.join("model", "saved_models", name_map[model_name])




def run_notebook(model_name):
    notebook_map = {
        "Logistic Regression": "Deb_ML_ASSN2_1_Logistic_Regression.ipynb",
        "Decision Tree": "Deb_ML_ASSN2_2_Decision_Tree.ipynb",
        "KNN": "Deb_ML_ASSN2_3_KNN.ipynb",
        "Naive Bayes": "Deb_ML_ASSN2_4_Naive_Bayes.ipynb",
        "Random Forest": "Deb_ML_ASSN2_5_Ensemble_Random_Forest.ipynb",
        "XGBoost": "Deb_ML_ASSN2_6_Ensemble_XGBoost.ipynb",
    }

    # # notebook_path = MODEL_DIR / notebook_map[model_name]
    # nb_path = os.path.join("model", notebook_map[model_name])
    
    # try:
    #     with open(nb_path, encoding='utf-8') as f:
    #         nb = nbformat.read(f, as_version=4)
        
    #     # This preprocessor executes the notebook and UPDATES the 'nb' object in place
    #     ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    #     ep.preprocess(nb, {'metadata': {'path': 'model/'}})
        
    #     return True, nb  # Return success and the notebook data
    # except Exception as e:
    #     return False, str(e)

    
    nb_path = os.path.join("model", notebook_map[model_name])
    
    try:
        # 1. Load the notebook
        with open(nb_path, encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)


        # 1. Configure the exporter to include ALL outputs
        c = Config()
        c.HTMLExporter.preprocessors = [
            'nbconvert.preprocessors.ExecutePreprocessor',
        ]
        
        # 2. Initialize with config
        html_exporter = HTMLExporter()
        html_exporter.exclude_input = True # Show only results, not code


        
        # 2. Execute the notebook
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        ep.preprocess(nb, {'metadata': {'path': 'model/'}})
        
        # 3. Convert the executed notebook to HTML
        (body, resources) = html_exporter.from_notebook_node(nb)
        
        return True, body
    except Exception as e:
        return False, str(e)
    

def run_notebook_to_html(model_name):

    notebook_map = {
        "Logistic Regression": "Deb_ML_ASSN2_1_Logistic_Regression.ipynb",
        "Decision Tree": "Deb_ML_ASSN2_2_Decision_Tree.ipynb",
        "KNN": "Deb_ML_ASSN2_3_KNN.ipynb",
        "Naive Bayes": "Deb_ML_ASSN2_4_Naive_Bayes.ipynb",
        "Random Forest": "Deb_ML_ASSN2_5_Ensemble_Random_Forest.ipynb",
        "XGBoost": "Deb_ML_ASSN2_6_Ensemble_XGBoost.ipynb",
    }
    nb_path = os.path.join("model", notebook_map[model_name])

    # 1. Check if the file actually exists first
    if not os.path.exists(nb_path):
        return False, f"Notebook not found at {nb_path}"
    
    try:

        # 1. Load the notebook
        with open(nb_path, encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)

        # 1. Configure the exporter to include ALL outputs
        c = Config()
        c.HTMLExporter.preprocessors = [
            'nbconvert.preprocessors.ExecutePreprocessor',
        ]
        
        # 2. Initialize with config
        html_exporter = HTMLExporter(config=c)
        html_exporter.exclude_input = True # Show only results, not code
        
        # 3. Re-run execution (ensure the notebook object is updated)
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        ep.preprocess(nb, {'metadata': {'path': 'model/'}})
        
        # 4. Export
        (body, resources) = html_exporter.from_notebook_node(nb)
        return True, body
        
    except Exception as e:
        return False, str(e)

    


# Helper function to display notebook outputs
def display_notebook_results(nb):
    st.info("### 📋 Notebook Execution Logs")
    
    for cell in nb.cells:
        if cell.cell_type == 'code':
            for output in cell.get('outputs', []):
                # 1. Handle standard text/logs
                if output.output_type == 'stream':
                    st.text(output.text)
                
                # 2. Handle Plots & Figures (The critical fix)
                elif output.output_type in ['display_data', 'execute_result']:
                    data = output.get('data', {})
                    
                    # Try PNG first (most common for Matplotlib/Seaborn)
                    if 'image/png' in data:
                        img_data = base64.b64decode(data['image/png'])
                        st.image(img_data, use_container_width=True)
                    
                    # Try JPEG
                    elif 'image/jpeg' in data:
                        img_data = base64.b64decode(data['image/jpeg'])
                        st.image(img_data, use_container_width=True)
                    
                    # Try SVG (Vector graphics)
                    elif 'image/svg+xml' in data:
                        st.write("*(SVG Plot)*")
                        st.components.v1.html(data['image/svg+xml'], scrolling=True)

                    # Handle HTML (Pandas tables or interactive plots)
                    elif 'text/html' in data:
                        st.components.v1.html(data['text/html'], height=300, scrolling=True)
                    
                    # Fallback for plain text results (like "Accuracy: 0.95")
                    elif 'text/plain' in data:
                        st.write(data['text/plain'])

                # 3. Handle Errors with full traceback
                elif output.output_type == 'error':
                    st.error(f"Execution Error: {output.ename}")
                    st.exception(Exception(output.evalue))




# --------------------------------------------------
# 1. TRAIN MODEL BY RUNNING NOTEBOOK
# --------------------------------------------------

def train_model(model_name):
    notebook_map = {
        "Logistic Regression": "Deb_ML_ASSN2_1_Logistic_Regression.ipynb",
        "Decision Tree": "Deb_ML_ASSN2_2_Decision_Tree.ipynb",
        "KNN": "Deb_ML_ASSN2_3_KNN.ipynb",
        "Naive Bayes": "Deb_ML_ASSN2_4_Naive_Bayes.ipynb",
        "Random Forest": "Deb_ML_ASSN2_5_Ensemble_Random_Forest.ipynb",
        "XGBoost": "Deb_ML_ASSN2_6_Ensemble_XGBoost.ipynb",
    }

    notebook_path = MODEL_DIR / notebook_map[model_name]

    subprocess.run(
        [
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            str(notebook_path),
        ],
        check=True,
    )


# --------------------------------------------------
# 2. LOAD MODEL
# --------------------------------------------------

def load_model(model_name):
    model_map = {
        "Logistic Regression": joblib.load("model/logistic_regression_model.pkl"),
        "Decision Tree": joblib.load("model/decision_tree_model.pkl"),
        "KNN": joblib.load("model/knn_model.pkl"),
        "Gaussian Naive Bayes": joblib.load("model/gaussian_nb_model.pkl"),
        "Random Forest": joblib.load("model/random_forest_model.pkl"),
        "XGBoost": joblib.load("model/xgboost_model.pkl"),
    }

    model = joblib.load(MODEL_DIR / model_map[model_name])
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")

    return model, scaler


# --------------------------------------------------
# 3. PREDICTION
# --------------------------------------------------

def predict(model_name, df):
    model, scaler = load_model(model_name)

    X = df.copy()

    if model_name in ["Logistic Regression", "KNN", "Naive Bayes"]:
        X = scaler.transform(X)

    preds = model.predict(X)
    df["prediction"] = preds

    output_path = DATA_DIR / "prediction_output.csv"
    df.to_csv(output_path, index=False)

    return df, output_path
