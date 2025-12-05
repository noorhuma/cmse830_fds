# 💉 Health Risk Prediction Dashboard

## 📌 Overview
This project is a **Streamlit-based interactive dashboard** for predicting risk of **Heart Disease**, **Diabetes**, and **Stroke**.  
It was developed as part of **CMSE 830: Foundations for Data Science** at Michigan State University.

The app allows users to:
- Explore datasets with summary statistics and visualizations
- Understand how missing data is handled
- Train and evaluate models (Logistic Regression, Random Forest)
- View performance metrics (Confusion Matrix, ROC Curve, Classification Report, Feature Importance)
- Enter personal details to get a personalized risk prediction

---

## 🚀 Features
- **Dataset Overview**: Shape, missing values, and class balance
- **Exploratory Data Analysis (EDA)**: Histograms, boxplots, violin plots, pairplots, correlation heatmaps, descriptive statistics
- **Missingness Handling**: Visualizes missing values and explains cleaning strategy (drop or impute)
- **Feature Engineering**: Derived feature `BMI_Category` (Underweight, Normal, Overweight, Obese)
- **Modeling**: Logistic Regression and Random Forest with preprocessing pipelines
- **Validation**: Train/test split, 5‑fold cross‑validation, GridSearchCV hyperparameter tuning for Random Forest
- **Evaluation Metrics**: Confusion Matrix, ROC Curve with AUC, Classification Report, Feature Importance
- **Interactive Prediction**: User-friendly form with dropdowns and number inputs for risk prediction
- **Download Option**: Export predictions as CSV (optional enhancement)

---

## 📊 Data Dictionary (Example: Diabetes Dataset)

| Column                  | Type     | Description                                      |
|--------------------------|----------|--------------------------------------------------|
| Pregnancies              | int      | Number of times pregnant                         |
| Glucose                  | float    | Plasma glucose concentration                     |
| BloodPressure            | float    | Diastolic blood pressure (mm Hg)                 |
| SkinThickness            | float    | Triceps skin fold thickness (mm)                 |
| Insulin                  | float    | 2-Hour serum insulin (mu U/ml)                   |
| BMI                      | float    | Body mass index (weight/height²)                 |
| DiabetesPedigreeFunction | float    | Genetic risk score                               |
| Age                      | int      | Age in years                                     |
| Outcome                  | int      | Target (1 = diabetes, 0 = no diabetes)           |
| BMI_Category             | category | Derived feature: Underweight / Normal / Overweight / Obese |

Similar dictionaries are provided for **Heart Disease** and **Stroke** datasets.

---

## 🧠 Modeling Approach
- **Preprocessing**:  
  - Numeric features scaled with `StandardScaler` (Logistic Regression only)  
  - Categorical features one‑hot encoded with `OneHotEncoder`  
- **Feature Engineering**: Added `BMI_Category` derived feature  
- **Models**:  
  - Logistic Regression (baseline linear model)  
  - Random Forest (ensemble model with hyperparameter tuning)  
- **Validation**:  
  - Train/test split (70/30 stratified)  
  - 5‑fold cross‑validation for robust evaluation  
  - GridSearchCV for Random Forest hyperparameter tuning  
- **Evaluation Metrics**:  
  - Confusion Matrix  
  - ROC Curve with AUC  
  - Classification Report (precision, recall, F1‑score)  
  - Feature Importance (Random Forest)

---

## 🛠️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/noorhuma/cmse830_fds.git
cd cmse830_fds
pip install -r requirements.txt

▶️ Running the App
streamlit run app/streamlit_app.py
https://cmse830fds-2bhdrzewhthtpjpeqr5kdd.streamlit.app/

📂 Project Structure
cmse830_fds/
│
├── app/
│   └── streamlit_app.py        # Main Streamlit app
├── data/                       # Datasets (heart.csv, diabetes.csv, stroke.csv)
├── requirements.txt            # Python dependencies
└── .streamlit/
    └── config.toml             # Theme configuration

👩‍💻 Author
Developed by Humaira Noor 
Graduate Student, 
Data Science 
Michigan State University
