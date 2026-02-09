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
from utils import run_notebook, get_model_path, train_model, predict, display_notebook_results, run_notebook_to_html


st.set_page_config(page_title="Mobile Price Classification", layout="wide")


# Page refresh button at top
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    if st.button("🔄 Reset/Refresh App", type="primary"):
        st.rerun()

st.markdown("---")

st.title("📱 Machine Learning Models (Mobile Price Classification)")

# --------------------------------------------------
# MODE SELECTION
# --------------------------------------------------
mode = st.radio(
    "Choose Action",
    ["Train a Model", "Predict a Model"],
    index=None
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

    st.header("Training a Model...")
    st.subheader("Train a Machine Learning Model")

    model_choice = st.selectbox(
        "Select a Model to Train",
        models,
        index=None,
        placeholder="Select a model"
    )

    if model_choice:
        if st.button("🚀 Train Model"):
            with st.spinner("Training model..."):
                success, html_content = run_notebook_to_html(model_choice)
            if success:
                st.success(f"✅ {model_choice} trained and saved successfully!")
                # Use an expander to show logs so they don't clutter the UI
                with st.expander("Click to view full Training Logs, Visualizations & Metrics"):
                    # display_notebook_results(result)
                    components.html(html_content, height=800, scrolling=True)
            else:
                st.error("❌ Training failed. Check notebook paths.")
                st.info(result)

# --------------------------------------------------
# PREDICT MODE
# --------------------------------------------------
else:
    pass