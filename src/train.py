# src/train.py
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from src.preprocess import build_preprocess


def train_and_save(df, numeric, categorical, target, out_path):
    """
    Train Logistic Regression and Random Forest models on the given dataset,
    save them as .pkl files, and return evaluation metrics plus evaluation data.
    """
    # Split features and target
    X = df[numeric + categorical]
    y = df[target]

    # Build preprocessing pipeline
    pre = build_preprocess(numeric, categorical)

    # Define models
    models = {
        "logreg": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "rf": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    }

    results = {}
    outputs = {}

    for name, model in models.items():
        pipe = Pipeline([("pre", pre), ("model", model)])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=0.2, random_state=42
        )
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        # Save model
        joblib.dump(pipe, f"{out_path}_{name}.pkl")

        # Store results
        results[name] = {"accuracy": acc, "f1": f1}
        outputs[name] = {
            "y_test": y_test,
            "preds": preds,
            "proba": pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None
        }

    return results, outputs
