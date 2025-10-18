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
    sex = st.radio("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
    chol = st.slider("Cholesterol (chol)", 100, 400, 200)
    trestbps = st.slider("Resting Blood Pressure (trestbps)", 90, 200, 120)
    thalach = st.slider("Max Heart Rate Achieved (thalach)", 60, 200, 150)

    sex_encoded = 1 if sex == "Male" else 0

    return pd.DataFrame([{
        "age": age,
        "sex": sex_encoded,
        "cp": cp,
        "chol": chol,
        "trestbps": trestbps,
        "thalach": thalach
    }])

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
    st.subheader("📌 Feature Importance")
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
    with st.sidebar:
        st.title("🩺 Dual Disease Predictor")
        page = st.radio("Choose a tab:", [
            "Project Overview", 
            "Heart Disease", 
            "Stroke Risk", 
            "Merged Dataset"
        ])
        with st.expander("ℹ️ About This App"):
            st.markdown("""
            **CMSE 830 Midterm Project**  
            Created by Humaira  
            Fall 2025

            This app predicts heart disease and stroke risk using real clinical data and machine learning models.
            """)

    heart_model, stroke_model = load_models()

    if page == "Project Overview":
        st.title("🩺 Dual Disease Risk Explorer")
        st.markdown("""
        ### 🎬 Opening Scene: A Shared Risk Landscape
        Cardiovascular disease is the leading cause of death globally, and stroke is one of its most devastating outcomes. 
        While often studied separately, these conditions share many risk factors — age, hypertension, cholesterol, and more.

        This app explores a powerful question:

        > **Can we build a unified tool that helps people understand their risk for both heart disease and stroke — and see how these risks overlap?**

        ---
        ### 🔍 What You’ll Find in This App:
        - **Heart Disease Predictor**: Input patient data and get a prediction
        - **Stroke Risk Predictor**: Explore stroke risk based on key features
        - **Merged Dataset Insights**: Visualize comorbidity and shared risk factors
        - **Feature Importance**: See what drives each model’s decisions

        ---
        ### 🧠 Why It Matters
        - Data science can bridge gaps between related conditions
        - Interactive tools make complex models accessible
        - Merged datasets reveal comorbid patterns that single-disease studies miss

        This project is a story of connection — between datasets, between diseases, and between people and their health.
        """)

    elif page == "Heart Disease":
        st.title("❤️ Heart Disease Classifier")
        st.markdown("### 🔍 Act I: Heart Disease Risk Modeling")
        st.markdown("This model uses features like age, sex, chest pain, and cholesterol to predict cardiac risk.")
        user_input = get_heart_inputs()
        if st.button("Predict Heart Disease"):
            if heart_model is not None:
                try:
                    prediction = heart_model.predict(user_input)[0]
                    st.success("Prediction: Disease" if prediction == 1 else "Prediction: No Disease")
                    plot_feature_importance(heart_model, user_input.columns)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
            else:
                st.warning("Heart model not loaded.")

    elif page == "Stroke Risk":
        st.title("🧠 Stroke Risk Predictor")
        st.markdown("### 🔍 Act II: Stroke Risk Modeling")
        st.markdown("This model focuses on hypertension, glucose levels, and BMI to assess stroke risk.")
        user_input = get_stroke_inputs()
        if st.button("Predict Stroke Risk"):
            if stroke_model is not None:
                try:
                    prediction = stroke_model.predict(user_input)[0]
                    st.success("Prediction: Stroke Risk" if prediction == 1 else "Prediction: Low Risk")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
            else:
                st.warning("Stroke model not loaded.")

    elif page == "Merged Dataset":
        st.title("📊 Merged Dataset Insights")
        st.markdown("### 🔗 Act III: Exploring Comorbidity")
        st.markdown("By merging the datasets, we explore how shared risk factors affect both diseases.")
        try:
            merged_df = pd.read_csv("data/merged_health_data.csv")

            st.subheader("📊 Age Distribution by Disease")
            st.bar_chart(merged_df.groupby("age")[["target", "stroke"]].sum())

            st.subheader("🔁 Patients with Both Conditions")
            overlap = merged_df[(merged_df["target"] == 1) & (merged_df["stroke"] == 1)]
            st.write(f"Patients with both conditions: {len(overlap)}")

            st.subheader("🧪 Correlation Heatmap of Shared Features")
            numeric_df = merged_df.select_dtypes(include="number").dropna()
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

        except FileNotFoundError:
            st.warning("⚠️ Merged dataset not found. Please upload 'data/merged_health_data.csv'.")

if __name__ == "__main__":
    main()
