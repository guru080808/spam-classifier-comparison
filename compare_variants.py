"""Compare Decision Tree and KNN settings using training data only.

The five fitted models used by app.py are not changed by this script.
Run with ``python compare_variants.py`` to regenerate model_variants.json.
"""

from __future__ import annotations

import json

import joblib

from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from train_models import ROOT, SEED, TARGET, calculate_metrics, load_source


def tree_candidates():
    """Include unpruned, pre-pruned, and cost-complexity-pruned trees."""
    settings = [
        ("unpruned", None, 1, 0.0),
        ("pre-pruned", 8, 5, 0.0),
        ("pre-pruned", 5, 10, 0.0),
        ("post-pruned", None, 1, 0.001),
        ("post-pruned", None, 1, 0.005),
    ]
    for criterion in ("gini", "entropy"):
        for pruning, depth, leaf, alpha in settings:
            params = {
                "criterion": criterion,
                "max_depth": depth,
                "min_samples_leaf": leaf,
                "ccp_alpha": alpha,
            }
            model = make_pipeline(
                SimpleImputer(strategy="median"),
                DecisionTreeClassifier(random_state=SEED, **params),
            )
            yield {"family": "Decision Tree", "pruning": pruning, **params}, model


def knn_candidates():
    """Compare neighborhood size, voting and Manhattan/Euclidean distance."""
    for neighbors in (3, 5, 7, 11):
        for weights in ("uniform", "distance"):
            for distance in ("Manhattan", "Euclidean"):
                p = 1 if distance == "Manhattan" else 2
                model = make_pipeline(
                    SimpleImputer(strategy="median"),
                    StandardScaler(),
                    KNeighborsClassifier(
                        n_neighbors=neighbors, weights=weights, p=p
                    ),
                )
                yield {
                    "family": "KNN",
                    "neighbors": neighbors,
                    "weights": weights,
                    "distance": distance,
                }, model


def main() -> None:
    frame, features = load_source()
    X_train, X_test, y_train, y_test = train_test_split(
        frame[features],
        frame[TARGET],
        test_size=0.20,
        stratify=frame[TARGET],
        random_state=SEED,
    )
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    metrics = {"f1": "f1", "mcc": make_scorer(matthews_corrcoef)}
    results = []

    for description, model in (*tree_candidates(), *knn_candidates()):
        scores = cross_validate(
            model,
            X_train,
            y_train,
            scoring=metrics,
            cv=folds,
            n_jobs=1,
        )
        results.append(
            {
                **description,
                "cv_f1_mean": float(scores["test_f1"].mean()),
                "cv_f1_std": float(scores["test_f1"].std()),
                "cv_mcc_mean": float(scores["test_mcc"].mean()),
            }
        )

    selected = {}
    for family in ("Decision Tree", "KNN"):
        group = [row for row in results if row["family"] == family]
        best = max(group, key=lambda row: row["cv_f1_mean"])
        if family == "Decision Tree":
            model = make_pipeline(
                SimpleImputer(strategy="median"),
                DecisionTreeClassifier(
                    criterion=best["criterion"],
                    max_depth=best["max_depth"],
                    min_samples_leaf=best["min_samples_leaf"],
                    ccp_alpha=best["ccp_alpha"],
                    random_state=SEED,
                ),
            )
            filename = "decision_tree_entropy_pruned.joblib"
        else:
            model = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                KNeighborsClassifier(
                    n_neighbors=best["neighbors"],
                    weights=best["weights"],
                    p=1 if best["distance"] == "Manhattan" else 2,
                ),
            )
            filename = "knn_tuned.joblib"
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        joblib.dump(model, ROOT / "model" / filename, compress=3)
        selected[family] = {
            "settings": best,
            "model_file": filename,
            "held_out_metrics": calculate_metrics(
                y_test, predictions, probabilities
            ),
        }

    output = {
        "selection_data": "training partition only (3680 rows)",
        "cross_validation": "three stratified folds, shuffled with random_state=135",
        "selection_metric": "mean F1 for spam (1)",
        "results": results,
        "selected_alternatives": selected,
    }
    path = ROOT / "model_variants.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved {len(results)} settings to {path.name}")
    for family, details in selected.items():
        best = details["settings"]
        held_out = details["held_out_metrics"]
        print(
            f"{family}: CV F1={best['cv_f1_mean']:.4f}, "
            f"held-out F1={held_out['F1']:.4f}; {best}"
        )


if __name__ == "__main__":
    main()
