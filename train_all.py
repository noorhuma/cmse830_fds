import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from src.evaluate import plot_confusion, plot_roc, plot_feature_importance

OUT_DIR = os.path.abspath(os.path.join(os.getcwd(), "outputs"))
os.makedirs(OUT_DIR, exist_ok=True)
print("Outputs directory:", OUT_DIR)

def save_plot(fig, filename):
    output_path = os.path.join(OUT_DIR, filename)
    try:
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"✅ Saved: {output_path} ({os.path.getsize(output_path)} bytes)")
        else:
            print(f"⚠️ File not saved correctly: {output_path}")
    except Exception as e:
        print(f"❌ Error saving {filename}: {e}")

def evaluate_model(model, X_test, y_test, feature_names, prefix):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    fig = plot_confusion(y_test, y_pred, title=f"{prefix} Confusion Matrix")
    save_plot(fig, f"{prefix}_confusion.png")

    fig = plot_roc(y_test, y_proba, title=f"{prefix} ROC Curve")
    save_plot(fig, f"{prefix}_roc.png")

    if hasattr(model, "feature_importances_"):
        fig = plot_feature_importance(model, feature_names, title=f"{prefix} Feature Importance")
        save_plot(fig, f"{prefix}_feature_importance.png")

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_proba),
    }

def run_all_datasets():
    metrics = {}
    datasets = {
        "heart": ("data/heart.csv", "target"),
        "diabetes": ("data/diabetes.csv", "Outcome"),
    }

    for name, (path, target_col) in datasets.items():
        print(f"\n=== Training on {name} dataset ===")

        df = pd.read_csv(path)

        # Drop rows with missing target
        df = df.dropna(subset=[target_col])

        # 🚨 Diagnostic: show NaN counts before cleaning
        print(f"NaN counts in raw {name} dataset:")
        print(df.isna().sum())

        # Special cleaning for stroke dataset
        if name == "stroke":
            if "smoking_status" in df.columns:
                df["smoking_status"] = df["smoking_status"].fillna("Unknown")
            if "bmi" in df.columns:
                df["bmi"] = df["bmi"].fillna(df["bmi"].median())
            # Drop any rows still containing NaNs
            before = len(df)
            df = df.dropna()
            after = len(df)
            print(f"Stroke dataset: dropped {before - after} rows with remaining NaNs")
            print("Remaining NaNs after cleaning:", df.isna().sum().sum())

        X = df.drop(columns=[target_col])
        y = df[target_col]

        # One-hot encode categorical features
        X_encoded = pd.get_dummies(X, drop_first=True)

        # Impute missing values (most frequent for categorical safety)
        imputer = SimpleImputer(strategy="most_frequent")
        X_clean = imputer.fit_transform(X_encoded)

        # Drop any rows that still contain NaNs
        X_clean_df = pd.DataFrame(X_clean, columns=X_encoded.columns)
        before_drop = len(X_clean_df)
        mask = ~X_clean_df.isna().any(axis=1)
        X_clean_df = X_clean_df[mask]
        after_drop = len(X_clean_df)
        if before_drop - after_drop > 0:
            print(f"Dropped {before_drop - after_drop} rows with remaining NaNs in {name} dataset")

        # Align y with cleaned X
        y = y.iloc[X_clean_df.index]

        # Convert back to numpy for training
        X_clean = X_clean_df.values
        feature_names = list(X_encoded.columns)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y, test_size=0.2, random_state=42, stratify=y
        )

        metrics[name] = {}

        # Scale features for Logistic Regression
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 🚨 Final scrub: replace any NaNs/Infs with 0
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        # Logistic Regression
        logreg = LogisticRegression(max_iter=5000)
        logreg.fit(X_train_scaled, y_train)
        metrics[name]["logreg"] = evaluate_model(
            logreg, X_test_scaled, y_test, feature_names, prefix=f"{name}_logreg"
        )

        # Random Forest (no scaling needed)
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_train, y_train)
        metrics[name]["rf"] = evaluate_model(
            rf, X_test, y_test, feature_names, prefix=f"{name}_rf"
        )

    metrics_path = os.path.join(OUT_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\n✅ Metrics saved to {metrics_path}")

if __name__ == "__main__":
    run_all_datasets()
