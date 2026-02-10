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
from utils import run_notebook, get_model_path, train_model, predict, display_notebook_results, run_notebook_to_html, convert_notebook_to_html, display_notebook


st.set_page_config(page_title="Mobile Price Classification", layout="wide")


# Page refresh button at top
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    if st.button("🔄 Reset/Refresh App", type="primary"):
        st.rerun()

st.markdown("---")

st.title("📱 Machine Learning Models (Mobile Price Classification)")
st.warning("Dear Sir/Madam, If the App fails, that might be envirnmental issue! Please Call/WhatsApp me at +91-9177762671. - Regards, Debayan.")

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

    st.header("Train Your Machine Learning Model...")

    model_choice = st.selectbox(
        "Select Your Model to Train",
        models,
        index=None,
        placeholder="Select a model"
    )

    if model_choice:

        # Use an expander to show logs so they don't clutter the UI
        with st.expander("Click here to view Full Pre-Trained Logs and Metrics with Visualizations & Plots"):
            # display_notebook_results(result)
            # components.html(html_content, height=800, scrolling=True)

            # nb_html = convert_notebook_to_html(model_choice)
            # components.html(nb_html, height=800, scrolling=True)
            
            # Display using a scrollable component
            display_notebook(model_choice)

        
        if st.button("🚀 or Freshly Train the Model"):
            with st.spinner("Training your model! Please wait for less than 59 Seconds..."):
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
else:
    pass