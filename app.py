"""Streamlit test-set evaluator for five saved Spambase classifiers."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix


ROOT = Path(__file__).resolve().parent
META = json.loads((ROOT / "model" / "metadata.json").read_text(encoding="utf-8"))


@st.cache_resource
def load_model(model_name: str):
    """Load a fitted pipeline; do not fit on the uploaded test set."""
    return joblib.load(ROOT / "model" / META["model_files"][model_name])


def validate_test_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Check feature/target schema before any prediction is made."""
    required = [*META["features"], META["target"]]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing {len(missing)} required columns: {', '.join(missing[:8])}")
    if frame.empty:
        raise ValueError("The test CSV has no data rows.")
    X = frame[META["features"]].apply(pd.to_numeric, errors="raise")
    y = pd.to_numeric(frame[META["target"]], errors="raise")
    if y.isna().any() or not set(y.unique()).issubset({0, 1}):
        raise ValueError("The is_spam target must contain only 0 and 1, without blanks.")
    if y.nunique() < 2:
        raise ValueError("AUC needs both classes (0 and 1) in the test CSV.")
    return X, y.astype(int)


def score(model, X: pd.DataFrame, y: pd.Series) -> tuple[dict, pd.Series]:
    """Evaluate one already-trained classifier on exactly these test rows."""
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    prediction = model.predict(X)
    probability = model.predict_proba(X)[:, 1]
    metrics = {
        "Accuracy": accuracy_score(y, prediction),
        "AUC": roc_auc_score(y, probability),
        "Precision": precision_score(y, prediction, zero_division=0),
        "Recall": recall_score(y, prediction, zero_division=0),
        "F1": f1_score(y, prediction, zero_division=0),
        "MCC": matthews_corrcoef(y, prediction),
    }
    return metrics, pd.Series(prediction, index=y.index, name="Predicted")


st.set_page_config(page_title="Spam classifier comparison", layout="wide")
st.title("Spam classifier comparison")
st.caption(
    "UCI Spambase · 57 features · 5 saved classifiers · positive class: spam (1)"
)

uploaded = st.file_uploader(
    "Upload a test CSV with all 57 feature columns and is_spam",
    type="csv",
)
try:
    test_frame = pd.read_csv(uploaded if uploaded is not None else ROOT / "test_data.csv")
    X_test, y_test = validate_test_data(test_frame)
except (OSError, ValueError, pd.errors.ParserError) as exc:
    st.error(f"Cannot evaluate this CSV: {exc}")
    st.stop()

st.write(
    f"Evaluating **{len(y_test)} rows** from "
    + ("uploaded CSV" if uploaded is not None else "the bundled held-out test_data.csv")
    + ". Models are loaded from disk; no training occurs here."
)
selected = st.selectbox("Choose a classifier", list(META["model_files"]))

rows = []
for name in META["model_files"]:
    metrics, _ = score(load_model(name), X_test, y_test)
    rows.append({"Model": name, **metrics})

comparison = pd.DataFrame(rows).set_index("Model")
st.subheader("Same test set, all five models")
st.dataframe(comparison.style.format("{:.4f}"), use_container_width=True)

selected_metrics, selected_prediction = score(load_model(selected), X_test, y_test)
st.subheader(selected)
left, right = st.columns(2)
with left:
    st.write("Confusion matrix (actual rows, predicted columns)")
    confusion = confusion_matrix(y_test, selected_prediction, labels=[0, 1])
    st.dataframe(
        pd.DataFrame(
            confusion,
            index=["Actual non-spam (0)", "Actual spam (1)"],
            columns=["Predicted non-spam (0)", "Predicted spam (1)"],
        ),
        use_container_width=True,
    )
with right:
    st.write("Selected-model metrics")
    st.dataframe(
        pd.DataFrame(selected_metrics.items(), columns=["Metric", "Value"])
        .set_index("Metric")
        .style.format("{:.4f}"),
        use_container_width=True,
    )

st.write("Classification report")
report = classification_report(
    y_test,
    selected_prediction,
    labels=[0, 1],
    target_names=["non-spam (0)", "spam (1)"],
    output_dict=True,
    zero_division=0,
)
st.dataframe(pd.DataFrame(report).transpose().style.format("{:.4f}"))
