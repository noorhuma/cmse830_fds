# 🩺 Dual Disease Modeling: Heart Disease & Stroke Risk Prediction

## 📚 CMSE 830-001 — Fall 2025  
**Instructor**: Dr. Silvestri  
**Student**: Humaira  
**Project Type**: Midterm — Streamlit App + GitHub Repository

---

## 🧠 Project Overview

This project explores two clinically related datasets to model cardiovascular risk through classification. By analyzing shared features such as age, hypertension, cholesterol, and smoking status, we build predictive models for both heart disease and stroke. The workflow includes data cleaning, exploratory data analysis (EDA), imputation, and interactive visualizations. A Streamlit app presents both models with user-friendly controls and insights.

---

## 📊 Datasets Used

1. **Cleveland Heart Disease Dataset**  
   - Source: [Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)  
   - Features: 14 clinical attributes (e.g., age, sex, cholesterol, chest pain type)  
   - Target: `target` (0 = no disease, 1 = disease)

2. **Stroke Prediction Dataset**  
   - Source: [Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)  
   - Features: age, hypertension, heart disease, glucose level, BMI, smoking status  
   - Target: `stroke` (0 = no stroke, 1 = stroke)

---

## 🧹 Data Preparation

- Removed duplicates and handled missing values (`bmi`, `smoking_status`)
- Encoded categorical variables (`sex`, `cp`, `smoking_status`, `work_type`)
- Normalized numerical features for consistent scaling
- Applied basic imputation (mean/mode) for missing values
- Saved cleaned datasets as `heart_cleaned.csv` and `stroke_cleaned.csv` in the `data/` folder

---

## 🔗 Merged Dataset

To compare shared risk factors across both diseases, a merged dataset was created using:
- Shared features: `age`, `sex`, `heart_disease`
- Targets: `target` (heart disease), `stroke` (stroke risk)

This merged dataset enables:
- Dual-risk visualizations
- Comorbidity analysis
- Unified Streamlit interface

Saved as: `merged_heart_stroke.csv`

---

## 📈 Exploratory Data Analysis

**Visualizations**:
- Correlation heatmaps
- Histograms and boxplots for feature distributions
- Scatter plots and bar charts for feature-target relationships

**Statistical Summaries**:
- Mean, median, standard deviation, and value counts
- Feature importance via model coefficients

---

## 🧪 Modeling Approach

- **Heart Disease**: Logistic Regression, Decision Tree, Random Forest  
- **Stroke**: Logistic Regression, Random Forest, Gradient Boosting

Model performance evaluated using:
- Accuracy, precision, recall, F1-score  
- Confusion matrix and ROC curves

Models saved as `.pkl` files for app integration.

---

## 🌐 Streamlit App

Deployed app includes:
- **Two interactive tabs**: Heart Disease Classifier & Stroke Risk Predictor  
- **User controls**: Feature sliders, dropdowns, and prediction buttons  
- **Visuals**: Real-time plots, model metrics, and feature importance

🔗 [App Link](https://cmse830fds-2bhdrzewhthtpjpeqr5kdd.streamlit.app/)

---

## ⚙️ Setup Instructions

> These steps assume you have Python 3.8+ installed and Git configured.

### 1. Clone the Repository

```bash
git clone https://github.com/noorhuma/cmse830_fds.git
cd cmse830_fds

## 📁 Repository Structure

cmse830_fds/
│
├── data/                         # Raw and cleaned datasets
│   ├── heart.csv
│   ├── stroke.csv
│   ├── heart_cleaned.csv
│   ├── stroke_cleaned.csv
│   └── merged_heart_stroke.csv
│
├── notebooks/                    # Jupyter notebooks for EDA and merging
│   ├── eda_heart.ipynb
│   ├── eda_stroke.ipynb
│   └── merge_eda.ipynb
│
├── app/                          # Streamlit app and model files
│   ├── streamlit_app.py
│   ├── heart_model.pkl
│   └── stroke_model.pkl
│
├── .venv/                        # Virtual environment folder (created locally)
│   └── ...                      # Scripts and site-packages (excluded from Git)
│
├── .gitignore                   # Git exclusions (e.g., .venv/, __pycache__)
├── requirements.txt             # Project dependencies
└── README.md                    # Project overview and setup instructions
