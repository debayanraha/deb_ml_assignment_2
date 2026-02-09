import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score,
    confusion_matrix, classification_report
)

import seaborn as sns
import matplotlib.pyplot as plt



def predict_logistic_regression(df):

    model = joblib.load("model/logistic_regression_model.pkl")
    models["Logistic Regression"] = model
    scaler = joblib.load("model/standard_scaler.pkl")

    st.success(f"The Selected Model: {model}")

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
    
    st.info("New Features Created:")
    st.info("="*80)
    new_features = ['screen_area', 'screen_size', 'total_camera_mp', 'feature_count', 
                    'battery_efficiency', 'performance_score']
    for feature in new_features:
        st.info(f"✓ {feature}")
    
    st.info(f"\nTotal Features: {df_engineered.shape[1] - 1} (Original: {df.shape[1] - 1}, New: {len(new_features)})")
    df_engineered.head()
    
    
    # Check for missing values
    st.info("Missing Values:")
    st.info("="*80)
    missing_values = df_engineered.isnull().sum()
    st.info(missing_values)
    st.info(f"\nTotal missing values: {missing_values.sum()}")
    
    if missing_values.sum() == 0:
        st.info("\n✓ No missing values found!")
        
    
    # Check for duplicate rows
    st.info("Duplicate Rows:")
    st.info("="*80)
    duplicates = df_engineered.duplicated().sum()
    st.info(f"Number of duplicate rows: {duplicates}")
    
    if duplicates == 0:
        st.info("\n✓ No duplicate rows found!")
    
    
    # X = df_engineered.drop('price_range', axis=1)
    # y = df_engineered['price_range']

    X = df_engineered.copy()
    
    # Scaling logic
    X_test_scaled = scaler.transform(X)
    
    
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)
    

    st.info("✓ Predictions completed!")
    st.info("="*80)
    st.info(f"Predictions shape: {y_pred.shape}")
    st.info(f"\nPrediction probabilities shape (test): {y_proba.shape}")
    st.info(f"(Each sample has 4 probabilities, one for each class)")
    
    # Sample predictions with probabilities
    st.info("Sample Predictions with Probabilities:")
    st.info("="*80)
    sample_df = pd.DataFrame({
        'Predicted': y_pred[:10],
        'Prob_Class_0': y_proba[:10, 0],
        'Prob_Class_1': y_proba[:10, 1],
        'Prob_Class_2': y_proba[:10, 2],
        'Prob_Class_3': y_proba[:10, 3]
    })
    st.info(sample_df)

    

    return


def predict_decision_tree(df):


    model = joblib.load("model/decision_tree_model.pkl")
    models["Decision Tree"] = model

    st.success(f"The Selected Model: {model}")

    return


def predict_knn(df):

    model = joblib.load("model/knn_model.pkl")
    models["KNN"] = model

    st.success(f"The Selected Model: {model}")

    return

    

def predict_naive_bayes(df):

    model = joblib.load("model/gaussian_nb_model.pkl")
    models["Gaussian Naive Bayes"] = model

    st.success(f"The Selected Model: {model}")

    return


    
def predict_random_forest(df):

    model = joblib.load("model/random_forest_model.pkl")
    models["Random Forest"] = model

    st.success(f"The Selected Model: {model}")

    return




def predict_xgboost(df):


    model = joblib.load("model/xgboost_model.pkl")
    models["XGBoost"] = model

    st.success(f"The Selected Model: {model}")

    return






st.set_page_config(page_title="Mobile Price Classification", layout="wide")


# --------------------------------------------------
# RESET BUTTON
# --------------------------------------------------
if st.button("🔄 Refresh / Reset App"):
    st.rerun()


st.title("📱 Machine Learning Model with Mobile Price Classification Dataset")



# Initialize session state
if "confirmed" not in st.session_state:
    st.session_state.confirmed = False


DATA_PATH = Path("data/mobile_price_classification_test.csv")

@st.cache_data
def load_local_csv(path):
    return pd.read_csv(path)

if DATA_PATH.exists():
    df = load_local_csv(DATA_PATH)
    st.success("Built-in Test Dataset loaded successfully, having first 5 rows as:")
    st.dataframe(df.head())
else:
    st.error("Dataset file not found in app folder.")


csv_bytes = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Download Built-in Test Dataset",
    data=csv_bytes,
    file_name="mobile_price_classification_test.csv",
    mime="text/csv"
)


st.subheader("Make a Prediction for Mobile Price")

source = st.radio(
    "Choose your test dataset source:",
    ["Predict using Built-in Test Dataset", "Upload your New Test Dataset"]
)

if source == "Predict using Built-in Test Dataset":
    df = load_local_csv(DATA_PATH)    
else:
    # Upload dataset
    uploaded_file = st.file_uploader("Upload Test Dataset", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

st.subheader("Dataset Preview")
st.dataframe(df.head())




# Load models
models = {
    "Logistic Regression": joblib.load("model/logistic_regression_model.pkl"),
    "Decision Tree": joblib.load("model/decision_tree_model.pkl"),
    "KNN": joblib.load("model/knn_model.pkl"),
    "Gaussian Naive Bayes": joblib.load("model/gaussian_nb_model.pkl"),
    "Random Forest": joblib.load("model/random_forest_model.pkl"),
    "XGBoost": joblib.load("model/xgboost_model.pkl"),
}


# Confirmation button
if st.button("✅ Confirm dataset & proceed to model selection"):
    st.session_state.confirmed = True

    
# Show selectbox ONLY after confirmation
if st.session_state.confirmed:
    st.subheader("Select Machine Learning Model for Prediction")
    model_choice = st.selectbox(
        "Select Model",
        options=["-- Select a model --"] + list(models.keys()),
        index=0
    )
    if model_choice == "-- Select a model --":
        st.warning("Please select a model to continue.")
    else:
        # st.success(f"Selected model: {model_choice}")
        if model_choice == "Logistic Regression":
            predict_logistic_regression(df)
            
        elif model_choice == "Decision Tree":
            predict_decision_tree(df)
    
        elif model_choice == "KNN":
            predict_knn(df)
    
        elif model_choice == "Gaussian Naive Bayes":
            predict_naive_bayes(df)
    
        elif model_choice == "Random Forest":
            predict_random_forest(df)
    
        elif model_choice == "XGBoost":
            predict_xgboost(df)
    
        else:
            raise ValueError(f"Unsupported model: {model_choice}")
        
    
else:
    st.info("Please confirm to proceed.")







