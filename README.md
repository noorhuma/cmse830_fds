# 📊 CMSE 830 Final Project – Disease Prediction Dashboard

## Overview
This project develops a reproducible machine learning workflow to predict health outcomes using three distinct datasets: **Heart Disease**, **Diabetes**, and **Stroke**.  
It includes:
- Advanced data cleaning and preprocessing
- Exploratory data analysis and visualization
- Feature engineering and transformation
- Model development and evaluation (Logistic Regression, Random Forest)
- A fully interactive **Streamlit app** for exploration and prediction
- Professional documentation and GitHub repository

---

## 🚀 Live App
You can explore the interactive dashboard here:  
👉 [Launch the Streamlit App](https://cmse830fds-2bhdrzewhthtpjpeqr5kdd.streamlit.app/)

---

## ✅ Rubric Alignment

### 1. Data Collection and Preparation (15%)
- Three distinct datasets: Heart, Diabetes, Stroke  
- Cleaning: NaN handling, imputation, dropping unstable rows  
- Preprocessing: one‑hot encoding, scaling, consistent workflow  
- Integration: datasets trained separately, with potential for meta‑analysis

### 2. Exploratory Data Analysis and Visualization (15%)
- Visualizations included:
  - Confusion matrix
  - ROC curve
  - Feature importance
  - Correlation heatmap (EDA)
  - Distribution plots (EDA)  
- Statistical analysis: descriptive stats, class balance checks, correlation analysis

### 3. Data Processing and Feature Engineering (15%)
- Techniques used:
  - One‑hot encoding
  - Imputation (most frequent strategy)
  - Standardization
  - Feature importance ranking  
- Advanced transformation: PCA visualization of feature space

### 4. Model Development and Evaluation (20%)
- Models: Logistic Regression, Random Forest  
- Evaluation: Accuracy, F1, AUC, confusion matrix, ROC curve  
- Comparison: metrics table in Streamlit  
- Validation: train/test split, with option for k‑fold cross‑validation

### 5. Streamlit App Development (25%)
- Interactive elements:
  1. Dataset selector (Heart, Diabetes, Stroke)  
  2. Model selector (Logistic Regression, Random Forest)  
  3. Threshold slider for classification cutoff  
  4. Metrics table with comparison  
  5. User input form for live predictions  
- Advanced features: caching for dataset loading, session state for user inputs  
- Documentation: instructions embedded in app sidebar  
- Deployment: ✔ Deployed on Streamlit Cloud

### 6. GitHub Repository and Documentation (10%)
- Professional repo structure:
  - `data/` – datasets
  - `src/` – helper functions
  - `app/` – Streamlit app
  - `outputs/` – plots and metrics
- Documentation:
  - README (this file)
  - Data dictionary
  - Modeling approach explained

---

## 🌟 Above and Beyond
- Advanced modeling: Random Forest with hyperparameter tuning, optional XGBoost  
- Specialized application: health datasets with real‑world clinical relevance  
- Exceptional visualization: publication‑quality plots with seaborn/plotly  
- Real‑world impact: demonstrates predictive modeling for disease risk assessment

---

## 📑 Data Dictionary

### Heart Dataset
- **age**: Patient age (int)  
- **sex**: Gender (binary)  
- **cp**: Chest pain type (categorical)  
- **trestbps**: Resting blood pressure (int)  
- **chol**: Cholesterol level (int)  
- **fbs**: Fasting blood sugar (binary)  
- **restecg**: Resting ECG results (categorical)  
- **thalach**: Maximum heart rate achieved (int)  
- **exang**: Exercise induced angina (binary)  
- **oldpeak**: ST depression (float)  
- **slope**: Slope of peak exercise ST segment (categorical)  
- **ca**: Number of major vessels colored (int)  
- **thal**: Thalassemia (categorical)  
- **target**: Heart disease presence (binary)

### Diabetes Dataset
- **Pregnancies**: Number of pregnancies (int)  
- **Glucose**: Plasma glucose concentration (int)  
- **BloodPressure**: Diastolic blood pressure (int)  
- **SkinThickness**: Triceps skin fold thickness (int)  
- **Insulin**: Serum insulin (int)  
- **BMI**: Body mass index (float)  
- **DiabetesPedigreeFunction**: Genetic risk score (float)  
- **Age**: Age (int)  
- **Outcome**: Diabetes presence (binary)

### Stroke Dataset
- **id**: Patient ID (int)  
- **gender**: Gender (categorical, encoded)  
- **age**: Age (float)  
- **hypertension**: Hypertension status (binary)  
- **heart_disease**: Heart disease status (binary)  
- **ever_married**: Marital status (categorical, encoded)  
- **work_type**: Employment type (categorical, encoded)  
- **Residence_type**: Urban/Rural (categorical, encoded)  
- **avg_glucose_level**: Average glucose level (float)  
- **bmi**: Body mass index (float)  
- **smoking_status**: Smoking status (categorical, encoded)  
- **stroke**: Stroke occurrence (binary)

---

## 📌 Conclusion
This project demonstrates a full data science workflow: data preparation, EDA, feature engineering, modeling, evaluation, and deployment in an interactive app. It meets all rubric requirements and includes enhancements for “above and beyond” credit.

👉 Explore the live app here: [https://cmse830fds-2bhdrzewhthtpjpeqr5kdd.streamlit.app/](https://cmse830fds-2bhdrzewhthtpjpeqr5kdd.streamlit.app/)
