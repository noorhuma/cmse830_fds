import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Dual Disease Predictor", layout="wide")

def load_models():
    try:
        heart_model = joblib.load("app/heart_model.pkl")
        stroke_model = joblib.load("app/stroke_model.pkl")
        return heart_model, stroke_model
    except FileNotFoundError:
        st.error("❌ Model files not found.")
        return None, None

def get_heart_inputs():
    st.subheader("Enter Patient Info for Heart Disease Prediction")
    age = st.slider("Age", 20, 80, 50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
    chol = st.slider("Cholesterol (chol)", 100, 400, 200)
    trestbps = st.slider("Resting Blood Pressure (trestbps)", 90, 200, 120)
    thalach = st.slider("Max Heart Rate Achieved (thalach)", 60, 200, 150)
    sex_encoded = 1 if sex == "Male" else 0
    return pd.DataFrame([[age, sex_encoded, cp, chol, trestbps, thalach]],
                        columns=["age", "sex", "cp", "chol", "trestbps", "thalach"])

def get_stroke_inputs():
    st.subheader("Enter Patient Info for Stroke Risk Prediction")
    age = st.slider("Age", 20, 80, 50)
    hypertension = st.selectbox("Hypertension", [0, 1])
    heart_disease = st.selectbox("Heart Disease", [0, 1])
    avg_glucose = st.slider("Average Glucose Level", 50.0, 250.0, 100.0)
    bmi = st.slider("BMI", 10.0, 50.0, 25.0)
    return pd.DataFrame([[age, hypertension, heart_disease, avg_glucose, bmi]],
                        columns=["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"])

def plot_feature_importance(model, feature_names):
    st.subheader("Feature Importance")
    try:
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(len(importances)), importances[indices])
        ax.set_xticks(range(len(importances)))
        ax.set_xticklabels([feature_names[i] for i in indices], rotation=45)
        ax.set_title("Feature Importance")
        st.pyplot(fig)
    except AttributeError:
        st.info("Feature importance not available for this model.")

def main():
    st.sidebar.title("🩺 Dual Disease Predictor")
    page = st.sidebar.radio("Choose a model:", ["Heart Disease", "Stroke Risk", "Merged Dataset"])

    heart_model, stroke_model = load_models()

    if page == "Heart Disease":
        st.title("❤️ Heart Disease Classifier")
        user_input = get_heart_inputs()
        if st.button("Predict Heart Disease"):
            prediction = heart_model.predict(user_input)[0]
            st.success("Prediction: Disease" if prediction == 1 else "Prediction: No Disease")
            plot_feature_importance(heart_model, user_input.columns)

    elif page == "Stroke Risk":
        st.title("🧠 Stroke Risk Predictor")
        user_input = get_stroke_inputs()
        if st.button("Predict Stroke Risk"):
            prediction = stroke_model.predict(user_input)[0]
            st.success("Prediction: Stroke Risk" if prediction == 1 else "Prediction: Low Risk")

    elif page == "Merged Dataset":
        st.title("📊 Merged Dataset Insights")
        try:
            merged_df = pd.read_csv("data/merged_health_data.csv")
            st.write("Explore shared risk factors across heart disease and stroke.")

            st.subheader("Age Distribution by Disease")
            st.bar_chart(merged_df.groupby("age")[["target", "stroke"]].sum())

            st.subheader("Comorbidity Overlap")
            overlap = merged_df[(merged_df["target"] == 1) & (merged_df["stroke"] == 1)]
            st.write(f"Patients with both conditions: {len(overlap)}")

            st.subheader("Correlation Heatmap")
            numeric_df = merged_df.select_dtypes(include="number").dropna()
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

        except FileNotFoundError:
            st.warning("⚠️ Merged dataset not found. Please upload 'data/merged_health_data.csv'.")

if __name__ == "__main__":
    main()
