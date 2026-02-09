import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from utils import train_model, predict


from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score,
    confusion_matrix, classification_report
)

import seaborn as sns
import matplotlib.pyplot as plt


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
                train_model(model_choice)
            st.success(f"{model_choice} trained and saved successfully!")

# --------------------------------------------------
# PREDICT MODE
# --------------------------------------------------
else:
    pass