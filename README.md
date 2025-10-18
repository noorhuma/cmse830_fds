# 🩺 Dual Disease Risk Explorer

A Streamlit web app that predicts **heart disease** and **stroke risk** using clinical data and machine learning — helping users explore each condition independently.

---

## 🎯 Project Overview

Cardiovascular disease is the leading cause of death globally, and stroke is one of its most devastating outcomes. While often studied together, this project takes a modular approach — offering separate tools to understand each risk clearly.

This project asks:

> **Can we build accessible tools that help people understand their risk for heart disease and stroke — side by side?**

---

## 📊 Features

- ❤️ **Heart Disease Predictor**  
  Input patient data and get a prediction using a Random Forest model trained on cardiac features

- 🧠 **Stroke Risk Predictor**  
  Explore stroke risk based on hypertension, glucose, and BMI

- 📌 **Feature Importance**  
  See what drives each model’s decisions

---

## 🧠 Models Used

| Disease        | Model(s) Used            | Features                                                  |
|----------------|--------------------------|-----------------------------------------------------------|
| Heart Disease  | Random Forest Classifier | age, sex, cp, chol, trestbps, thalach                     |
| Stroke Risk    | Random Forest Classifier | age, hypertension, heart_disease, avg_glucose_level, bmi |

Models were trained on cleaned versions of the [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease) and [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset).

---

## 📂 Folder Structure


Models were trained on cleaned versions of the [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease) and [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset).



cmse830_fds/ 
├── app/ 
│ ├── streamlit_app.py 
│ ├── heart_model.pkl 
│ └── stroke_model.pkl 
├── data/ 
│ ├── heart.csv 
│ └── stroke.csv 
├── notebooks/ 
│ └── modeling.ipynb 
└── README.md


---

## 🚀 How to Run

### 🔗 [Launch the App on Streamlit](https://cmse830fds-2bhdrzewhthtpjpeqr5kdd.streamlit.app/)

### Or run locally:

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
