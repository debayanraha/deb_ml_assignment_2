import subprocess
import joblib
import pandas as pd
from pathlib import Path


MODEL_DIR = Path("model")
DATA_DIR = Path("data")


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
