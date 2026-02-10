import streamlit as st
import streamlit.components.v1 as components
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
from utils import run_notebook, get_model_path, train_model, predict_model
from utils import display_notebook_results, run_notebook_to_html
from utils import convert_notebook_to_html, display_notebook


st.set_page_config(page_title="Mobile Price Classification", layout="wide")

# Initialize session state
if "confirmed" not in st.session_state:
    st.session_state.confirmed = False



# Page refresh button at top
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    if st.button("🔄 Reset/Refresh App", type="primary"):
        st.session_state.confirmed = True
        for key in ["action_mode"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

st.markdown("---")

st.title("📱 Machine Learning Models (Mobile Price Classification)")
st.warning("Dear Sir/Madam, If the App fails, that might be envirnmental issue! Please Call/WhatsApp me at +91-9177762671. - Regards, Debayan.")

# --------------------------------------------------
# MODE SELECTION
# --------------------------------------------------
mode = st.radio(
    "Choose Action",
    ["Train a Model", 
     "Predict a Model", 
     "View Evaluation Metrics",
     "View Confusion Matrix",
     "View Classification Report"
    ],
    index=None,
    key="action_mode"
)

st.markdown("---")

models = [
    "Logistic Regression",
    "Decision Tree",
    "KNN",
    "Naive Bayes",
    "Random Forest",
    "XGBoost",
]

# --------------------------------------------------
# TRAIN MODE
# --------------------------------------------------
if mode == "Train a Model":

    st.header("Train Your Machine Learning Model...")

    model_choice = st.selectbox(
        "Select Your Model to Train",
        models,
        index=None,
        placeholder="Select a model"
    )

    if model_choice:

        # Use an expander to show logs so they don't clutter the UI
        with st.expander("Click here to view Pre-Trained Full Logs and Metrics with Visualizations & Plots"):
            # display_notebook_results(result)
            # components.html(html_content, height=800, scrolling=True)

            # nb_html = convert_notebook_to_html(model_choice)
            # components.html(nb_html, height=800, scrolling=True)
            
            # Display using a scrollable component
            display_notebook(model_choice)

        
        if st.button("🚀 or Freshly Train the Model again"):
            with st.spinner("Training your model! Wait time not more than 59 Seconds..."):
                success, html_content = run_notebook_to_html(model_choice)
            if success:
                st.success(f"✅ {model_choice} trained and saved successfully!")

                # Use an expander to show logs so they don't clutter the UI
                with st.expander("Click here to view CURRENT Training Logs & Metrics (without Visualizations)"):
                    # display_notebook_results(result)
                    components.html(html_content, height=800, scrolling=True)
                    
            else:
                st.error("❌ Training failed. Check notebook paths.")
                st.info(html_content)

# --------------------------------------------------
# PREDICT MODE
# --------------------------------------------------
elif mode == "Predict a Model":

    st.header("Predict Mobile Price using Trained Models...")

    TEST_IN_DATA_PATH = Path("data/mobile_price_classification_test.csv")
    TEST_OUT_DATA_PATH = Path("data/mobile_price_classification_test_prediction.csv")

    @st.cache_data
    def load_local_csv(path):
        return pd.read_csv(path)
    
    if TEST_IN_DATA_PATH.exists():
        df = load_local_csv(TEST_IN_DATA_PATH)
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
    
    
    st.subheader("Use Built-in Test Dataset or Upload a Custom Test Dataset")
    
    source = st.radio(
        "select your preference:",
        ["Predict using above Built-in Test Dataset", "Upload your New Test Dataset"]
    )
    
    if source == "Predict using Built-in Test Dataset":
        df = load_local_csv(TEST_IN_DATA_PATH)    
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
        "Naive Bayes": joblib.load("model/gaussian_nb_model.pkl"),
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
            if st.button("🔮 Predict"):
                with st.spinner("Running prediction..."):

                    df = df.drop(['id', 'price_range'], axis=1, errors='ignore')
                    outdf = predict_model(model_choice, models[model_choice], df)
                    
                    st.success("Prediction completed!")
                    outdf.to_csv(TEST_OUT_DATA_PATH, index=False)
                    st.dataframe(outdf.head())

                    st.download_button(
                    "⬇️ Download Full Mobile Price Prediction Output",
                    outdf.to_csv(index=False).encode("utf-8"),
                    "mobile_price_classification_test_prediction.csv",
                    "text/csv",
        )
        
    else:
        st.info("Please confirm to proceed.")


# --------------------------------------------------
# Show Evaluation Metrics
# --------------------------------------------------
elif mode == "View Evaluation Metrics":

    st.header("Evaluation Metrics of Models...")

    evaluation_metric = {
        "Logistic Regression": "model/logistic_regression_evaluation_metrics.csv",
        "Decision Tree": "model/decision_tree_evaluation_metrics.csv",
        "KNN": "model/knn_evaluation_metrics.csv",
        "Naive Bayes": "model/gaussian_nb_evaluation_metrics.csv",
        "Random Forest": "model/random_forest_evaluation_metrics.csv",
        "XGBoost": "model/xgboost_evaluation_metrics.csv",
    }
    

    model_choice = st.selectbox(
        "Select Model to view Evaluation Metrics",
        models,
        index=None,
        placeholder="Select a model"
    )

    if model_choice:

        METRIC_PATH = Path(evaluation_metric[model_choice])
    
        @st.cache_data
        def load_local_csv(path):
            return pd.read_csv(path)
        
        if METRIC_PATH.exists():
            df = load_local_csv(METRIC_PATH)
            st.dataframe(df)
        else:
            st.error("evaluation_metric file not found in app folder.")


# --------------------------------------------------
# View Confusion Matrix
# --------------------------------------------------
elif mode == "View Confusion Matrix":

    st.header("Confusion Matrix of Models...")

    confusion_matrix= {
        "Logistic Regression": "model/logistic_regression_confusion_matrix.csv",
        "Decision Tree": "model/decision_tree_confusion_matrix.csv",
        "KNN": "model/knn_confusion_matrix.csv",
        "Naive Bayes": "model/gaussian_nb_confusion_matrix.csv",
        "Random Forest": "model/random_forest_confusion_matrix.csv",
        "XGBoost": "model/xgboost_confusion_matrix.csv",
    }
    

    model_choice = st.selectbox(
        "Select Model to view Confusion Matrix",
        models,
        index=None,
        placeholder="Select a model"
    )

    if model_choice:

        CONFN_PATH = Path(confusion_matrix[model_choice])
    
        @st.cache_data
        def load_local_csv(path):
            return pd.read_csv(path)
        
        if CONFN_PATH.exists():
            df = load_local_csv(CONFN_PATH)
            st.dataframe(df)
        else:
            st.error("Confusion Matrix file not found in app folder.")



# --------------------------------------------------
# View Classification Report
# --------------------------------------------------
elif mode == "View Classification Report":

    st.header("Classification Report of Models...")

    confusion_matrix= {
        "Logistic Regression": "model/logistic_regression_classification_report",
        "Decision Tree": "model/decision_tree_classification_report",
        "KNN": "model/knn_classification_report",
        "Naive Bayes": "model/gaussian_nb_classification_report",
        "Random Forest": "model/random_forest_classification_report",
        "XGBoost": "model/xgboost_classification_report",
    }
    

    model_choice = st.selectbox(
        "Select Model to view Classification Report",
        models,
        index=None,
        placeholder="Select a model"
    )

    if model_choice:

        CLSSN_PATH = Path(confusion_matrix[model_choice])

        if CLSSN_PATH.exists():
            with open(CLSSN_PATH, "r") as f:
                report_txt = f.read()
            st.text(report_txt)
        else:
            st.error("Classification Report file not found in app folder.")
    