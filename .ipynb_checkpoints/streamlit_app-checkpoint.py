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


st.set_page_config(page_title="Mobile Price Classification", layout="wide")





# Page refresh button at top
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("🔄 Reset/Refresh App", type="primary"):
        st.rerun()

st.markdown("---")

st.title("📱 Machine Learning Models (Mobile Price Classification)")



