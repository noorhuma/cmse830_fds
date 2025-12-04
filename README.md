📊 CMSE 830 Final Project – Disease Prediction Dashboard
Overview
This project develops a reproducible machine learning workflow to predict health outcomes using three distinct datasets: Heart Disease, Diabetes, and Stroke. It includes:

Advanced data cleaning and preprocessing

Exploratory data analysis and visualization

Feature engineering and transformation

Model development and evaluation (Logistic Regression, Random Forest)

A fully interactive Streamlit app for exploration and prediction

Professional documentation and GitHub repository

✅ Rubric Alignment
1. Data Collection and Preparation (15%)
Three distinct datasets: Heart, Diabetes, Stroke

Cleaning: NaN handling, imputation, dropping unstable rows

Preprocessing: one‑hot encoding, scaling, consistent workflow

Integration: datasets trained separately, with potential for meta‑analysis

2. Exploratory Data Analysis and Visualization (15%)
Visualizations included:

Confusion matrix

ROC curve

Feature importance

Correlation heatmap (EDA)

Distribution plots (EDA)

Statistical analysis: descriptive stats, class balance checks, correlation analysis

3. Data Processing and Feature Engineering (15%)
Techniques used:

One‑hot encoding

Imputation (most frequent strategy)

Standardization

Feature importance ranking

Advanced transformation: PCA visualization of feature space (added for rubric)

4. Model Development and Evaluation (20%)
Models: Logistic Regression, Random Forest

Evaluation: Accuracy, F1, AUC, confusion matrix, ROC curve

Comparison: metrics table in Streamlit

Validation: train/test split, with option for k‑fold cross‑validation

5. Streamlit App Development (25%)
Interactive elements:

Dataset selector (Heart, Diabetes, Stroke)

Model selector (Logistic Regression, Random Forest)

Threshold slider for classification cutoff

Metrics table with comparison

User input form for live predictions

Advanced features: caching for dataset loading, session state for user inputs

Documentation: instructions embedded in app sidebar

Deployment: ready for Streamlit Cloud

6. GitHub Repository and Documentation (10%)
Professional repo structure:

data/ – datasets

src/ – helper functions

app/ – Streamlit app

outputs/ – plots and metrics

Documentation:

README (this file)

Data dictionary (see below)

Modeling approach explained

🌟 Above and Beyond (20% potential)
Advanced modeling: Random Forest with hyperparameter tuning, optional XGBoost

Specialized application: health datasets with real‑world clinical relevance

Exceptional visualization: publication‑quality plots with seaborn/plotly

Real‑world impact: demonstrates predictive modeling for disease risk assessment