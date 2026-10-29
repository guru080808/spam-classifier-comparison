"""Train and evaluate five classifiers on the same UCI Spambase split.

Run once with ``python train_models.py``. The Streamlit app only loads the
saved pipelines; it never trains a model on uploaded test rows.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parent
SEED = 135
TARGET = "is_spam"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "gaussian_nb.joblib",
    "Support Vector Machine": "svm.joblib",
}


def load_source() -> tuple[pd.DataFrame, list[str]]:
    """Read the original UCI data and its official attribute names."""
    names_path = ROOT / "data" / "spambase.names"
    data_path = ROOT / "data" / "spambase.data"
    if not names_path.exists() or not data_path.exists():
        raise FileNotFoundError("Unzip the official spambase.zip into data/ first.")

    features = []
    for line in names_path.read_text(encoding="utf-8").splitlines():
        if ":" in line and line.strip().endswith("continuous."):
            features.append(line.split(":", 1)[0].strip())
    if len(features) != 57:
        raise ValueError(f"Expected 57 feature names; found {len(features)}")

    frame = pd.read_csv(data_path, header=None, names=[*features, TARGET])
    if frame.shape != (4601, 58):
        raise ValueError(f"Unexpected source shape: {frame.shape}")
    if set(frame[TARGET].unique()) != {0, 1}:
        raise ValueError("Target must contain both classes, encoded 0 and 1.")
    return frame, features


def build_models() -> dict[str, object]:
    """Keep preprocessing inside each saved pipeline to avoid test leakage."""
    return {
        "Logistic Regression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=SEED),
        ),
        "Decision Tree": make_pipeline(
            SimpleImputer(strategy="median"),
            DecisionTreeClassifier(
                max_depth=8, min_samples_leaf=5, random_state=SEED
            ),
        ),
        "K-Nearest Neighbors": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=7),
        ),
        "Gaussian Naive Bayes": make_pipeline(
            SimpleImputer(strategy="median"), GaussianNB()
        ),
        "Support Vector Machine": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            SVC(kernel="rbf", C=1.0, probability=True, random_state=SEED),
        ),
    }


def calculate_metrics(y_true: pd.Series, predictions, probabilities) -> dict:
    """Compute six evaluation metrics using spam (1) as positive."""
    return {
        "Accuracy": accuracy_score(y_true, predictions),
        "AUC": roc_auc_score(y_true, probabilities),
        "Precision": precision_score(y_true, predictions, zero_division=0),
        "Recall": recall_score(y_true, predictions, zero_division=0),
        "F1": f1_score(y_true, predictions, zero_division=0),
        "MCC": matthews_corrcoef(y_true, predictions),
    }


def main() -> None:
    frame, features = load_source()
    X_train, X_test, y_train, y_test = train_test_split(
        frame[features],
        frame[TARGET],
        test_size=0.20,
        stratify=frame[TARGET],
        random_state=SEED,
    )
    # This is the only data the app evaluates by default. It is never fitted.
    X_test.assign(**{TARGET: y_test}).to_csv(ROOT / "test_data.csv", index=False)
    model_dir = ROOT / "model"
    model_dir.mkdir(exist_ok=True)

    results = []
    for name, pipeline in build_models().items():
        pipeline.fit(X_train, y_train)
        predicted = pipeline.predict(X_test)
        probability = pipeline.predict_proba(X_test)[:, 1]
        row = {"Model": name, **calculate_metrics(y_test, predicted, probability)}
        results.append(row)
        joblib.dump(pipeline, model_dir / MODEL_FILES[name], compress=3)
        print(f"{name}: F1={row['F1']:.4f}, MCC={row['MCC']:.4f}")

    pd.DataFrame(results).to_csv(ROOT / "evaluation.csv", index=False)
    metadata = {
        "source": "https://archive.ics.uci.edu/dataset/94/spambase",
        "target": TARGET,
        "features": features,
        "seed": SEED,
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "positive_class": 1,
        "model_files": MODEL_FILES,
    }
    (ROOT / "model" / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("Saved models, evaluation.csv, and test_data.csv")


if __name__ == "__main__":
    main()
