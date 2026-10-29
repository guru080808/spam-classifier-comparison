# Spam classifier comparison

Repository: https://github.com/guru080808/spam-classifier-comparison

## Problem statement

Classify an email as spam (1) or non-spam (0) from numerical features of its
word frequencies, character frequencies, and capital-letter runs. Five
classifiers are trained on one dataset and compared on the *same* untouched
test split. The app loads the saved models and evaluates a CSV; it does not
train on that CSV.

## Dataset

The [UCI Spambase dataset](https://archive.ics.uci.edu/dataset/94/spambase)
contains 4,601 email records, 57 input features, and one binary target. The
original UCI `spambase.data` and `spambase.names` are retained under `data/`.
The source is credited to Mark Hopkins, Erik Reeber, George Forman, and Jaap
Suermondt (1999), DOI [10.24432/C53G6X](https://doi.org/10.24432/C53G6X).
The UCI page states a CC BY 4.0 licence. Each input row describes an email;
the positive class (1) means spam.

## Method

- The source is read with the official 57 attribute names.
- A stratified 80/20 train–test split uses `random_state=135`: 3,680 training
  rows and 921 held-out test rows. `test_data.csv` is precisely that test set.
- Median imputation is fitted on training data *inside* every pipeline. A
  standard scaler is also fitted inside the Logistic Regression, KNN and SVM
  pipelines. Thus test values do not affect preprocessing parameters.
- Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes and an RBF SVM
  are trained once and saved in `model/`. `app.py` only loads them.
- AUC uses the spam-class probability. Precision, recall and F1 treat spam as
  the positive class. MCC uses both classes.

## Comparison on the held-out test set

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9218 | **0.9704** | 0.9422 | 0.8540 | 0.8960 | 0.8362 |
| Decision Tree | 0.9186 | 0.9327 | 0.9114 | 0.8788 | 0.8948 | 0.8288 |
| K-Nearest Neighbors | 0.8969 | 0.9538 | 0.9161 | 0.8127 | 0.8613 | 0.7832 |
| Gaussian Naive Bayes | 0.8198 | 0.9454 | 0.7014 | **0.9449** | 0.8052 | 0.6690 |
| Support Vector Machine | **0.9283** | 0.9700 | **0.9433** | 0.8705 | **0.9054** | **0.8497** |

The full-precision values are in `evaluation.csv`. Results can be regenerated
with `python train_models.py` from the original source files.

## Observations and overall winner

- **Logistic Regression:** strongest AUC (0.9704) and high precision, but
  slightly lower recall and F1 than the SVM.
- **Decision Tree:** good recall and interpretable splits, but the lowest AUC
  of the five models (0.9327).
- **KNN:** reasonable AUC; lower spam recall (0.8127) reduces F1.
- **Gaussian Naive Bayes:** highest recall (0.9449), but many more false
  positives are implied by its lower precision (0.7014).
- **SVM:** highest accuracy, precision, F1 and MCC, with AUC very close to
  Logistic Regression. It is the best overall choice here when false positives
  and false negatives both matter.

These are observations on one fixed test split, not claims that the SVM will
always win on every dataset or random split.

## Decision Tree and KNN settings

`compare_variants.py` compares settings using three stratified folds of the
3,680-row training partition. Mean spam-class F1 selects one setting per model
family. The 921-row test set is used only after that selection. The app uses
the five models in the main comparison table.

| Model setting | Training-fold F1 | Held-out F1 | Held-out MCC |
|---|---:|---:|---:|
| Decision Tree: Gini, depth 8, leaf 5 (main model) | 0.8841 | 0.8948 | 0.8288 |
| Decision Tree: entropy, `ccp_alpha=0.001` | **0.9041** | **0.9050** | **0.8449** |
| KNN: 7 neighbors, uniform, Euclidean (main model) | 0.8673 | 0.8613 | 0.7832 |
| KNN: 11 neighbors, distance-weighted, Euclidean | **0.9000** | **0.8911** | **0.8266** |

For trees, the comparison covers Gini versus entropy, unrestricted trees,
pre-pruning (`max_depth`, `min_samples_leaf`), and post-pruning (`ccp_alpha`).
For KNN, it covers 3, 5, 7 and 11 neighbors; uniform and distance-weighted
voting; and Manhattan and Euclidean distance. All 26 settings and their
cross-validation scores are in `model_variants.json`. The two selected
alternatives are saved separately in `model/`.

## Repository files

```text
app.py                 Streamlit evaluation interface
train_models.py        Source loading, split, training, metrics, artifact save
compare_variants.py    Training-fold comparisons for tree and KNN settings
requirements.txt       Python package versions
test_data.csv          Held-out rows, all 57 features and target is_spam
evaluation.csv         Six metrics for each of the five models
model_variants.json     Setting-level scores and selected alternative results
data/spambase.data     Original UCI dataset
data/spambase.names    Official UCI feature names
model/*.joblib         Five fitted preprocessing-and-classifier pipelines
                       plus two separately named alternatives
model/metadata.json    Feature order, target, split and model filenames
```

## Run locally

From this folder, with Python 3.10 or later:

```powershell
python -m pip install -r requirements.txt
python train_models.py
python compare_variants.py
python -m streamlit run app.py
```

Open the local URL shown by Streamlit. The app initially displays the bundled
`test_data.csv`. The upload control accepts a CSV with all 57 official feature
columns and `is_spam` containing both labels 0 and 1. Extra columns are
ignored.
