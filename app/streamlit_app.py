import streamlit as st
import pandas as pd
import json

# Page setup
st.set_page_config(page_title="CMSE 830 Project", layout="wide")
st.title("CMSE 830 Project: Health Prediction Models")

# Sidebar navigation
dataset = st.sidebar.radio("Select Dataset", ["Heart", "Stroke", "Diabetes"])

# Sidebar descriptions
st.sidebar.markdown("### Dataset Descriptions")
st.sidebar.markdown("""
- **Heart**: Predicts presence of heart disease based on clinical features.
- **Stroke**: Predicts stroke risk using demographic and health indicators.
- **Diabetes**: Predicts diabetes diagnosis based on physiological measurements.
""")

# Load metrics from JSON file
try:
    with open("outputs/metrics.json", "r") as f:
        metrics = json.load(f)
except FileNotFoundError:
    metrics = {}

def show_metrics(dataset_name):
    """Display metrics table for a dataset if available."""
    if dataset_name in metrics:
        df = pd.DataFrame(metrics[dataset_name]).T
        st.subheader("Model Metrics")
        st.dataframe(df.style.format({"Accuracy":"{:.2f}", "F1":"{:.2f}", "AUC":"{:.2f}"}))
    else:
        st.info("Metrics not found. Run train_all.py to generate metrics.")

def show_images(prefix):
    """Display saved plots for a dataset."""
    st.image(f"outputs/{prefix}_logreg_confusion.png", caption="Logistic Regression - Confusion Matrix")
    st.image(f"outputs/{prefix}_logreg_roc.png", caption="Logistic Regression - ROC Curve")
    st.image(f"outputs/{prefix}_rf_confusion.png", caption="Random Forest - Confusion Matrix")
    st.image(f"outputs/{prefix}_rf_roc.png", caption="Random Forest - ROC Curve")
    st.image(f"outputs/{prefix}_rf_feature_importance.png", caption="Random Forest - Feature Importance")

# Main content
if dataset == "Heart":
    st.header("Heart Dataset Results")
    show_metrics("heart")
    show_images("heart")

elif dataset == "Stroke":
    st.header("Stroke Dataset Results")
    show_metrics("stroke")
    show_images("stroke")

elif dataset == "Diabetes":
    st.header("Diabetes Dataset Results")
    show_metrics("diabetes")
    show_images("diabetes")
