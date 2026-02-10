import streamlit as st
import streamlit.components.v1 as components
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

    # notebook_path = MODEL_DIR / notebook_map[model_name]
    nb_path = os.path.join("model", notebook_map[model_name])
    
    try:
        with open(nb_path, encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        # This preprocessor executes the notebook and UPDATES the 'nb' object in place
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        ep.preprocess(nb, {'metadata': {'path': 'model/'}})
        
        return True, nb  # Return success and the notebook data
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

        # 2. Initialize with config
        html_exporter = HTMLExporter(template_name='basic')
        html_exporter.exclude_input = True # Show only results, not code
        
        # 3. Re-run execution (ensure the notebook object is updated)
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        ep.preprocess(nb, {'metadata': {'path': 'model/'}})
        
        # 4. Export
        (body, resources) = html_exporter.from_notebook_node(nb)
        return True, body
        
    except Exception as e:
        return False, str(e)



def convert_notebook_to_html(model_name):
    """
    Runs the terminal command: jupyter nbconvert --to html notebook.ipynb
    """

    notebook_map = {
        "Logistic Regression": "Deb_ML_ASSN2_1_Logistic_Regression.ipynb",
        "Decision Tree": "Deb_ML_ASSN2_2_Decision_Tree.ipynb",
        "KNN": "Deb_ML_ASSN2_3_KNN.ipynb",
        "Naive Bayes": "Deb_ML_ASSN2_4_Naive_Bayes.ipynb",
        "Random Forest": "Deb_ML_ASSN2_5_Ensemble_Random_Forest.ipynb",
        "XGBoost": "Deb_ML_ASSN2_6_Ensemble_XGBoost.ipynb",
    }
    notebook_path = os.path.join("model", notebook_map[model_name])

    # 1. Check if the file actually exists first
    if not os.path.exists(notebook_path):
        return False, f"Notebook not found at {notebook_path}"
    
    # 1. Define the input path and expected output path
    html_output_path = notebook_path.replace(".ipynb", ".html")
    
    try:
        # 2. Execute the shell command
        # --execute forces the notebook to run and generate plots before converting
        subprocess.run([
            "jupyter", "nbconvert", 
            "--to", "html", 
            "--execute", 
            "--no-input", # Optional: hides the code cells
            notebook_path
        ], check=True)
        
        # 3. Read the generated HTML file
        with open(html_output_path, "r", encoding="utf-8") as f:
            html_content = f.read()
            
        return html_content
    
    except subprocess.CalledProcessError as e:
        return f"Error during conversion: {e}"
    except FileNotFoundError:
        return "Error: HTML file was not generated. Check notebook execution."



def display_notebook(model_name):

    notebook_map = {
        "Logistic Regression": "Deb_ML_ASSN2_1_Logistic_Regression.html",
        "Decision Tree": "Deb_ML_ASSN2_2_Decision_Tree.html",
        "KNN": "Deb_ML_ASSN2_3_KNN.html",
        "Naive Bayes": "Deb_ML_ASSN2_4_Naive_Bayes.html",
        "Random Forest": "Deb_ML_ASSN2_5_Ensemble_Random_Forest.html",
        "XGBoost": "Deb_ML_ASSN2_6_Ensemble_XGBoost.html",
    }
    notebook_path = os.path.join("model", notebook_map[model_name])
    # 1. Check if the file actually exists first
    if not os.path.exists(notebook_path):
        return False, f"Notebook not found at {notebook_path}"
    
    # Load the pre-converted HTML file
    with open(notebook_path, 'r', encoding='utf-8') as f:
        html_data = f.read()
    
    # Display using a scrollable component
    components.html(html_data, height=800, scrolling=True)



    

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

def perform_pre_processing(df):
    
    # Create a copy for feature engineering
    df_engineered = df.copy()
    
    # 1. Screen Area (px_height * px_width)
    df_engineered['screen_area'] = df_engineered['px_height'] * df_engineered['px_width']
    
    # 2. Screen Size (sc_h * sc_w)
    df_engineered['screen_size'] = df_engineered['sc_h'] * df_engineered['sc_w']
    
    # 3. Camera Quality (fc + pc)
    df_engineered['total_camera_mp'] = df_engineered['fc'] + df_engineered['pc']
    
    # 4. Feature Count (sum of binary features)
    df_engineered['feature_count'] = (df_engineered['blue'] + df_engineered['dual_sim'] + 
                                      df_engineered['four_g'] + df_engineered['three_g'] + 
                                      df_engineered['touch_screen'] + df_engineered['wifi'])
    
    # 5. Battery Efficiency (battery_power / mobile_wt)
    df_engineered['battery_efficiency'] = df_engineered['battery_power'] / (df_engineered['mobile_wt'] + 1)
    
    # 6. Performance Score (ram * n_cores * clock_speed)
    df_engineered['performance_score'] = df_engineered['ram'] * df_engineered['n_cores'] * df_engineered['clock_speed']
    
    # # 7. Memory-to-Weight Ratio
    # df_engineered['memory_weight_ratio'] = df_engineered['int_memory'] / (df_engineered['mobile_wt'] + 1)
    
    # # 8. Pixel Density (approximation)
    # df_engineered['pixel_density'] = df_engineered['screen_area'] / (df_engineered['screen_size'] + 1)
    
    # # 9. Battery per Talk Time
    # df_engineered['battery_per_talk'] = df_engineered['battery_power'] / (df_engineered['talk_time'] + 1)
    
    # # 10. Average Camera MP
    # df_engineered['avg_camera_mp'] = (df_engineered['fc'] + df_engineered['pc']) / 2
    
    print("New Features Created:")
    # new_features = ['screen_area', 'screen_size', 'total_camera_mp', 'feature_count', 
    #                 'battery_efficiency', 'performance_score', 'memory_weight_ratio', 
    #                 'pixel_density', 'battery_per_talk', 'avg_camera_mp']
    new_features = ['screen_area', 'screen_size', 'total_camera_mp', 'feature_count', 
                'battery_efficiency', 'performance_score', 'memory_weight_ratio', 
                'pixel_density', 'battery_per_talk', 'avg_camera_mp']

    for feature in new_features:
        print(f"✓ {feature}")
    
    print(f"\nTotal Features: {df_engineered.shape[1] - 1} (Original: {df.shape[1] - 1}, New: {len(new_features)})")

    
    # Check for missing values
    missing_values = df_engineered.isnull().sum()
    st.info(f"\nTotal missing values: {missing_values.sum()}")
    
    if missing_values.sum() == 0:
        st.info("\n✓ No missing values found!")
    
    # Check for duplicate rows
    duplicates = df_engineered.duplicated().sum()
    st.info(f"Number of duplicate rows: {duplicates}")
    
    if duplicates == 0:
        st.info("\n✓ No duplicate rows found!")
    
    return df_engineered



def predict_model(model_choice, model, df):

    st.success(f"The Selected Model: {model}")

    X = perform_pre_processing(df)

    if model_choice == "Logistic Regression":
        
        # Scaling logic
        scaler = joblib.load("model/standard_scaler.pkl")
        X_test_scaled = scaler.transform(X)

    elif model_choice == "KNN":
        
        # Scaling logic
        scaler = joblib.load("model/knn_scaler.pkl")
        X_test_scaled = scaler.transform(X)

    else:
        X_test_scaled = X.copy()

    
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)

    preds = model.predict(X)
    df["prediction"] = y_pred
    df["Prob_Class_0"] = y_proba[:, 0]
    df["Prob_Class_1"] = y_proba[:, 1]
    df["Prob_Class_2"] = y_proba[:, 2]
    df["Prob_Class_3"] = y_proba[:, 3]

    return df


