import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="Health Risk Prediction", page_icon="💉", layout="wide")

# -----------------------------
# Sidebar navigation
# -----------------------------
st.sidebar.title("Navigation")
disease = st.sidebar.radio("Select disease", ["Heart Disease", "Diabetes", "Stroke"])
model_choice = st.sidebar.selectbox("Choose model", ["Logistic Regression", "Random Forest"])
threshold = st.sidebar.slider("Classification threshold", 0.1, 0.9, 0.5, 0.05)

st.sidebar.markdown("---")
st.sidebar.write("Steps:\n1. Select disease\n2. Explore dataset\n3. See models\n4. Enter details for prediction")

# -----------------------------
# Home page
# -----------------------------
st.title("💉 Health Risk Prediction Dashboard")
st.markdown("""
Welcome! This app predicts risk for **Heart Disease**, **Diabetes**, and **Stroke**.  
Use the sidebar to select a condition, explore the dataset, see how missing data is handled, 
and run models with clear visual explanations.  
Finally, enter your own details to check your risk.
""")
st.info("👉 Select a disease from the sidebar to begin.")

# -----------------------------
# Dataset loader with caching
# -----------------------------
@st.cache_data
def load_dataset(name):
    if name == "Heart Disease":
        df = pd.read_csv("data/heart.csv")
        target = "target"
    elif name == "Diabetes":
        df = pd.read_csv("data/diabetes.csv")
        target = "Outcome"
    elif name == "Stroke":
        df = pd.read_csv("data/stroke.csv")
        target = "stroke"
    else:
        df, target = None, None
    return df, target

# -----------------------------
# Dataset overview
# -----------------------------
df, target_col = load_dataset(disease)

# Derived feature: BMI category (only if BMI column exists)
if df is not None and "BMI" in df.columns:
    st.info("Creating derived feature: BMI category")
    def categorize_bmi(bmi):
        if bmi < 18.5:
            return "Underweight"
        elif bmi < 25:
            return "Normal"
        elif bmi < 30:
            return "Overweight"
        else:
            return "Obese"
    df["BMI_Category"] = df["BMI"].apply(categorize_bmi)
    df["BMI_Category"] = df["BMI_Category"].astype("category")  # Arrow-compatible
    st.write("Added BMI_Category column based on BMI values.")

st.header(f"{disease} Dataset Overview")
st.info("This section shows dataset shape, missing values, and class balance.")

if df is not None:
    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

    st.subheader("Missing values per column")
    st.write(df.isnull().sum())

    st.subheader("Class balance")
    st.bar_chart(df[target_col].value_counts())

    # -----------------------------
    # EDA Visualizations
    # -----------------------------
    st.header(f"{disease} – Exploratory Data Analysis")
    st.info("EDA plots help visualize distributions and correlations.")

    # Distribution plots
    st.subheader("Feature distributions")
    numeric_cols_preview = df.select_dtypes(include=['int64','float64']).columns[:4]
    for col in numeric_cols_preview:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax, color="skyblue")
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)

    # Correlation heatmap
    st.subheader("Correlation heatmap")
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(corr, cmap="Blues", ax=ax)
    st.pyplot(fig)

    # Violin plot
    num_cols_full = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    st.subheader("Violin plot by target")
    if len(num_cols_full) > 0 and target_col in df.columns:
        feature_for_violin = num_cols_full[0]
        try:
            fig, ax = plt.subplots()
            sns.violinplot(
                data=df,
                x=target_col,
                y=feature_for_violin,
                inner="quartile",
                color="skyblue"   # avoid palette+hue deprecation
            )
            ax.set_title(f"Violin plot of {feature_for_violin} by {target_col}")
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Violin plot skipped due to error: {e}")

    # Pairplot
    st.subheader("Pairplot (sampled columns)")
    sample_cols = num_cols_full[:5]
    if target_col in df.columns and target_col not in sample_cols:
        sample_cols = sample_cols + [target_col]
    try:
        fig = sns.pairplot(df[sample_cols], hue=target_col if target_col in sample_cols else None, diag_kind="kde")
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Pairplot skipped due to size or data type constraints: {e}")

    # Descriptive statistics
    st.subheader("Descriptive statistics")
    st.write(df.describe(include='all'))

    # Boxplots
    st.subheader("Boxplots")
    boxplot_cols = num_cols_full[:4]
    for col in boxplot_cols:
        fig, ax = plt.subplots()
        sns.boxplot(x=df[col], ax=ax, color="#2E86C1")
        ax.set_title(f"Boxplot of {col}")
        st.pyplot(fig)

else:
    st.warning("Dataset not found. Please check file paths.")

# -----------------------------
# Missingness Handling
# -----------------------------
st.header(f"{disease} – Handling Missing Data")
st.info("Missingness handling ensures clean data for modeling.")

if df is not None:
    cleaned_df = df.dropna()
    st.write(f"After dropping missing values: {cleaned_df.shape[0]} rows remain.")

# -----------------------------
# Modeling and evaluation
# -----------------------------
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import numpy as np

st.header(f"{disease} – Modeling and Evaluation")
st.info("Model evaluation shows confusion matrix, ROC curve, and metrics.")

if df is not None:
    # Prepare data
    cleaned_df = df.dropna()
    X = cleaned_df.drop(columns=[target_col])
    y = cleaned_df[target_col]

    # Train/test split FIRST
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Column types
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    # Preprocessing
    num_transformer = StandardScaler() if model_choice == "Logistic Regression" else "passthrough"
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    # Pipeline
    if model_choice == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
        clf = Pipeline(steps=[("pre", preprocessor), ("model", model)])
    else:
        model = RandomForestClassifier(random_state=42)
        clf = Pipeline(steps=[("pre", preprocessor), ("model", model)])

        # Hyperparameter tuning
        st.subheader("Hyperparameter tuning (Random Forest)")
        param_grid = {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [None, 5, 10],
            "model__min_samples_split": [2, 5, 10]
        }
        grid_search = GridSearchCV(clf, param_grid, cv=3, scoring="accuracy", n_jobs=-1)
        grid_search.fit(X_train, y_train)
        st.write("Best parameters found:", grid_search.best_params_)
        st.write(f"Best CV accuracy: {grid_search.best_score_:.3f}")
        clf = grid_search.best_estimator_

    # Fit final model
    clf.fit(X_train, y_train)

    # Predictions
    y_pred_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob >= threshold).astype(int)

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y_test, y_pred))

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color="blue", label=f"ROC curve (AUC = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    st.pyplot(fig)

    # Classification report
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

    # Cross-validation
    st.subheader("Cross-validation results")
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    st.write(f"Cross-validation accuracy scores: {cv_scores}")
    st.write(f"Mean CV accuracy: {cv_scores.mean():.3f}")

    # Feature importance (RF only; uses the tuned internal model)
    if model_choice == "Random Forest":
        try:
            # Extract underlying RF from pipeline
            rf_model = clf.named_steps["model"]
            # Build feature names after preprocessing
            ohe = clf.named_steps["pre"].named_transformers_["cat"]
            cat_feature_names = list(ohe.get_feature_names_out(cat_cols))
            feature_names = numeric_cols + cat_feature_names

            importances = rf_model.feature_importances_
            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False)
            st.subheader("Feature Importance")
            st.bar_chart(importance_df.set_index("Feature"))
        except Exception as e:
            st.warning(f"Feature importance unavailable: {e}")

# -----------------------------
# Interactive prediction
# -----------------------------
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

st.header(f"{disease} – Interactive Prediction")
st.info("Interactive prediction lets you test risk with your own inputs.")

if df is not None:
    st.subheader("Enter your details")

    cleaned_df = df.dropna()
    X = cleaned_df.drop(columns=[target_col])
    y = cleaned_df[target_col]

    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    num_transformer = StandardScaler() if model_choice == "Logistic Regression" else "passthrough"
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    if model_choice == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    else:
        model = RandomForestClassifier(n_estimators=200, random_state=42)

    clf_pred = Pipeline(steps=[("pre", preprocessor), ("model", model)])
    clf_pred.fit(X, y)

    user_input = {}
    for col in cat_cols:
        options = sorted([str(o) for o in X[col].dropna().unique()])
        user_input[col] = st.selectbox(f"{col}", options)

    for col in numeric_cols:
        col_min, col_max, col_med = float(X[col].min()), float(X[col].max()), float(X[col].median())
        if pd.api.types.is_integer_dtype(X[col]):
            user_input[col] = st.number_input(f"{col}", min_value=int(col_min), max_value=int(col_max), value=int(col_med), step=1)
        else:
            user_input[col] = st.number_input(f"{col}", min_value=col_min, max_value=col_max, value=col_med)

    input_df = pd.DataFrame([user_input])

    if st.button("Predict risk"):
        prob = clf_pred.predict_proba(input_df)[0][1]
        pred_class = "At Risk" if prob >= threshold else "Not at Risk"

        st.subheader("Prediction result")
        st.metric(label="Risk probability", value=f"{prob:.2f}")
        if pred_class == "At Risk":
            st.error(f"Model prediction: **{pred_class}** (threshold = {threshold:.2f})")
        else:
            st.success(f"Model prediction: **{pred_class}** (threshold = {threshold:.2f})")

        with st.expander("View your inputs"):
            st.write(input_df)
