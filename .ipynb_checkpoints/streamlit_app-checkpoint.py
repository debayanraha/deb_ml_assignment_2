import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score,
    confusion_matrix, classification_report
)

import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Mobile Price Classification", layout="wide")

st.title("📱 Mobile Price Classification – ML Models Demo")

# Upload dataset
uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

# Load models
models = {
    "Logistic Regression": joblib.load("model/logistic_regression_model.pkl"),
    "Decision Tree": joblib.load("model/decision_tree_model.pkl"),
    "KNN": joblib.load("model/knn_model.pkl"),
    # "Naive Bayes": joblib.load("model/naive_bayes.pkl"),
    # "Random Forest": joblib.load("model/random_forest.pkl"),
    # "XGBoost": joblib.load("model/xgboost.pkl"),
}

scaler = joblib.load("model/logistic_regression_standard_scaler.pkl")

model_choice = st.selectbox("Select Model", list(models.keys()))

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    X = df.drop("price_range", axis=1)
    y = df["price_range"]

    # Scaling logic
    if model_choice in ["Logistic Regression", "KNN", "Naive Bayes"]:
        X_input = scaler.transform(X)
    else:
        X_input = X

    model = models[model_choice]

    y_pred = model.predict(X_input)
    y_proba = model.predict_proba(X_input)

    st.subheader("📊 Evaluation Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{accuracy_score(y, y_pred):.4f}")
    col2.metric("Precision", f"{precision_score(y, y_pred, average='weighted'):.4f}")
    col3.metric("Recall", f"{recall_score(y, y_pred, average='weighted'):.4f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("F1 Score", f"{f1_score(y, y_pred, average='weighted'):.4f}")
    col5.metric("MCC", f"{matthews_corrcoef(y, y_pred):.4f}")
    col6.metric("AUC", f"{roc_auc_score(y, y_proba, multi_class='ovr'):.4f}")

    # Confusion Matrix
    st.subheader("🔍 Confusion Matrix")
    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # Classification Report
    st.subheader("📄 Classification Report")
    st.text(classification_report(y, y_pred))
