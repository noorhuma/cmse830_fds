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

# -------------------------------
# Setup outputs directory
# -------------------------------
OUT_DIR = os.path.abspath(os.path.join(os.getcwd(), "outputs"))
os.makedirs(OUT_DIR, exist_ok=True)
print("Outputs directory:", OUT_DIR)

# -------------------------------
# Helper functions
# -------------------------------
def save_plot(fig, filename):
    """Save matplotlib figure safely to outputs/"""
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
    """Evaluate model and save plots + metrics"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Confusion matrix
    fig = plot_confusion(y_test, y_pred, title=f"{prefix} Confusion Matrix")
    save_plot(fig, f"{prefix}_confusion.png")

    # ROC curve
    fig = plot_roc(y_test, y_proba, title=f"{prefix} ROC Curve")
    save_plot(fig, f"{prefix}_roc.png")

    # Feature importance (RF only)
    if hasattr(model, "feature_importances_"):
        fig = plot_feature_importance(model, feature_names, title=f"{prefix} Feature Importance")
        save_plot(fig, f"{prefix}_feature_importance.png")

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_proba),
    }

# -------------------------------
# Main training loop
# -------------------------------
def run_all_datasets():
    metrics = {}
    datasets = {
        "heart": ("data/heart.csv", "target"),
        "diabetes": ("data/diabetes.csv", "Outcome"),
        "stroke": ("data/stroke.csv", "stroke"),   # ✅ stroke always included
    }
    print("Datasets configured keys:", list(datasets.keys()))

    # 👉 Diagnostic print to confirm keys
    print("Datasets configured keys:", list(datasets.keys()))

    for name, (path, target_col) in datasets.items():
        print(f"\n=== Training on {name} dataset ===")

        # Load dataset
        df = pd.read_csv(path)
        print(f"{name} dataset shape: {df.shape}")
        print(f"{name} columns: {list(df.columns)}")
        print(f"NaN counts in raw {name} dataset:\n{df.isna().sum()}")

        df = df.dropna(subset=[target_col])

        # Split features/target
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # ✅ Stroke is already encoded, so skip get_dummies
        if name == "stroke":
            X_encoded = X.copy()
        else:
            X_encoded = pd.get_dummies(X, drop_first=True)

        # Impute missing values
        imputer = SimpleImputer(strategy="most_frequent")
        X_clean = imputer.fit_transform(X_encoded)

        # Convert back to DataFrame
        X_clean_df = pd.DataFrame(X_clean, columns=X_encoded.columns)
        mask = ~X_clean_df.isna().any(axis=1)
        X_clean_df = X_clean_df[mask]
        y = y.iloc[X_clean_df.index]

        X_clean = X_clean_df.values
        feature_names = list(X_encoded.columns)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y, test_size=0.2, random_state=42, stratify=y
        )

        metrics[name] = {}

        # Logistic Regression
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        logreg = LogisticRegression(max_iter=5000)
        logreg.fit(X_train_scaled, y_train)
        metrics[name]["logreg"] = evaluate_model(
            logreg, X_test_scaled, y_test, feature_names, prefix=f"{name}_logreg"
        )

        # Random Forest
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_train, y_train)
        metrics[name]["rf"] = evaluate_model(
            rf, X_test, y_test, feature_names, prefix=f"{name}_rf"
        )

    # Save metrics
    metrics_path = os.path.join(OUT_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\n✅ Metrics saved to {metrics_path}")

    # -------------------------------
    # Diagnostic check for stroke outputs
    # -------------------------------
    print("\nChecking outputs after training...")
    expected_files = [
        "stroke_logreg_confusion.png",
        "stroke_logreg_roc.png",
        "stroke_rf_confusion.png",
        "stroke_rf_roc.png",
        "stroke_rf_feature_importance.png"
    ]
    for fname in expected_files:
        fpath = os.path.join(OUT_DIR, fname)
        if os.path.exists(fpath):
            print(f"✅ Found {fname} ({os.path.getsize(fpath)} bytes)")
        else:
            print(f"⚠️ Missing {fname}")

# -------------------------------
# Entry point
# -------------------------------
if __name__ == "__main__":
    run_all_datasets()
