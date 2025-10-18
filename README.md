# 🩺 Dual Disease Risk Explorer

A Streamlit web app that predicts **heart disease** and **stroke risk** using clinical data and machine learning — and explores how these risks overlap.

---

## 🎯 Project Overview

Cardiovascular disease is the leading cause of death globally, and stroke is one of its most devastating outcomes. While often studied separately, these conditions share many risk factors — age, hypertension, cholesterol, and more.

This project asks:

> **Can we build a unified tool that helps people understand their risk for both heart disease and stroke — and see how these risks overlap?**

---

## 📊 Features

- 🔍 **Heart Disease Predictor**  
  Input patient data and get a prediction using a Random Forest model

- 🧠 **Stroke Risk Predictor**  
  Explore stroke risk based on hypertension, glucose, and BMI

- 🔗 **Merged Dataset Insights**  
  Visualize comorbidity patterns and shared risk factors

- 📌 **Feature Importance**  
  See what drives each model’s decisions

---

## 🧠 Models Used

| Disease        | Model(s) Used                        | Features |
|----------------|--------------------------------------|----------|
| Heart Disease  | Random Forest Classifier             | age, sex, cp, chol, trestbps, thalach |
| Stroke Risk    | Random Forest Classifier             | age, hypertension, heart_disease, avg_glucose_level, bmi |

Models were trained on cleaned versions of the [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease) and [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset).

---

## 📂 Folder Structure

cmse830_fds/ 
├── app/ 
│ ├── streamlit_app.py 
│ ├── heart_model.pkl 
│ └── stroke_model.pkl 
├── data/ 
│ ├── heart.csv 
│ ├── stroke.csv 
│ └── merged_health_data.csv
├── notebooks/ 
│ ├── modeling.ipynb 
│ └── merge_eda.ipynb 
└── README.md


---

## 🚀 How to Run

### 🔗 [Launch the App on Streamlit](https://cmse830fds-2bhdrzewhthtpjpeqr5kdd.streamlit.app/)

### Or run locally:

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
