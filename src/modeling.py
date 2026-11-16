"""Shared models and evaluation protocol for both classification tasks.

The cancellation task (train.py) and the card-fraud task (card_fraud.py) use
the four algorithms named in the thesis application -- logistic regression,
SVM, Random Forest and XGBoost -- under one identical procedure:

1. stratified 80/20 train/test split with a fixed seed (`split_data`);
2. hyperparameters from tuning.py (RandomizedSearchCV on the training split
   only), or the defaults in `build_model` when tuning has not been run;
3. repeated stratified 5-fold cross-validation on the training split, so
   every score is reported as mean +/- standard deviation (`repeated_cv`);
4. the test split is touched once, for the final evaluation.

Keeping all of this in one module is what makes the two tasks comparable.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.utils.metaestimators import available_if
from xgboost import XGBClassifier

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Categories rarer than this share of the training rows are merged into one
# "infrequent" level (mainly the ~170 countries, most with a handful of rows).
RARE_CATEGORY_SHARE = 0.01

# Kernel SVM training time grows roughly quadratically to cubically with the
# number of rows, so the SVM is trained on at most this many rows. Measured on
# this project: about 15 s for 20,000 rows, versus hours for the full data.
SVM_MAX_SAMPLES = 20_000

MODEL_NAMES = ["logistic_regression", "svm", "random_forest", "xgboost"]
MODEL_LABELS = {
    "logistic_regression": "Logistička regresija",
    "svm": "SVM",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "baseline": "Referentni model (većinska klasa)",
}


class SubsampledClassifier(ClassifierMixin, BaseEstimator):
    """Fit the wrapped classifier on at most `max_samples` training rows.

    Subsampling happens inside fit(), so cross-validation and hyperparameter
    search subsample each training fold only, while predictions are always
    made for every row.

    strategy="stratified"  keeps the class ratio (used for cancellations);
    strategy="undersample" keeps every positive row and fills the rest with
    negatives (used for card fraud: at 0.17% positives a stratified sample of
    20,000 rows would keep only about 30 frauds).
    """

    def __init__(self, estimator=None, max_samples=SVM_MAX_SAMPLES,
                 strategy="stratified", random_state=RANDOM_STATE):
        self.estimator = estimator
        self.max_samples = max_samples
        self.strategy = strategy
        self.random_state = random_state

    def _sample_indices(self, y: np.ndarray) -> np.ndarray:
        n = len(y)
        if self.max_samples is None or n <= self.max_samples:
            return np.arange(n)
        if self.strategy == "stratified":
            keep, _ = train_test_split(np.arange(n), train_size=self.max_samples,
                                       stratify=y, random_state=self.random_state)
        elif self.strategy == "undersample":
            rng = np.random.RandomState(self.random_state)
            positives = np.flatnonzero(y == 1)
            negatives = np.flatnonzero(y != 1)
            n_negatives = min(max(self.max_samples - len(positives), 0), len(negatives))
            keep = np.concatenate([positives, rng.choice(negatives, n_negatives, replace=False)])
        else:
            raise ValueError(f"Unknown strategy {self.strategy!r}")
        return np.sort(keep)

    def fit(self, X, y):
        y = np.asarray(y)
        keep = self._sample_indices(y)
        X_sub = X.iloc[keep] if hasattr(X, "iloc") else X[keep]
        self.estimator_ = clone(self.estimator).fit(X_sub, y[keep])
        self.classes_ = self.estimator_.classes_
        self.n_samples_fit_ = len(keep)
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    @available_if(lambda self: hasattr(self.estimator, "decision_function"))
    def decision_function(self, X):
        return self.estimator_.decision_function(X)

    @available_if(lambda self: hasattr(self.estimator, "predict_proba"))
    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


def split_data(X, y):
    """The one train/test split used by every script, so test sets match."""
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)


def build_preprocessor(numeric_cols, categorical_cols=()) -> ColumnTransformer:
    """Standardise numeric columns, one-hot encode categorical ones.

    Rare categories are merged inside the pipeline, so the rule is learned from
    the training rows only and applied unchanged to new bookings. Scaling is
    applied for every model: it is required by logistic regression and SVM
    and has no effect on tree-based models.
    """
    transformers = [("num", StandardScaler(), list(numeric_cols))]
    if len(categorical_cols):
        transformers.append((
            "cat",
            OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=RARE_CATEGORY_SHARE),
            list(categorical_cols),
        ))
    return ColumnTransformer(transformers)


def positive_class_weight(y) -> float:
    """Negatives per positive -- XGBoost's equivalent of class_weight='balanced'."""
    y = np.asarray(y)
    return float((y == 0).sum() / (y == 1).sum())


def build_model(name: str, task: str, y_train, params: dict | None = None, n_jobs: int = -1):
    """Unfitted classifier with this project's defaults, overridden by `params`."""
    if name == "logistic_regression":
        model = LogisticRegression(solver="liblinear", C=1.0, class_weight="balanced",
                                   max_iter=5000, random_state=RANDOM_STATE)
    elif name == "svm":
        model = SubsampledClassifier(
            SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", cache_size=1000),
            strategy="undersample" if task == "card" else "stratified",
        )
    elif name == "random_forest":
        depth, leaf, trees = (12, 3, 200) if task == "card" else (20, 5, 300)
        model = RandomForestClassifier(n_estimators=trees, max_depth=depth, min_samples_leaf=leaf,
                                       class_weight="balanced", n_jobs=n_jobs,
                                       random_state=RANDOM_STATE)
    elif name == "xgboost":
        model = XGBClassifier(
            n_estimators=400,
            max_depth=5 if task == "card" else 6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=positive_class_weight(y_train) if task == "card" else 1.0,
            eval_metric="aucpr" if task == "card" else "logloss",
            n_jobs=n_jobs,
            random_state=RANDOM_STATE,
        )
    else:
        raise ValueError(f"Unknown model {name!r}")

    if params:
        model.set_params(**params)
    return model


def build_pipeline(name, task, numeric_cols, categorical_cols, y_train,
                   params: dict | None = None, n_jobs: int = -1) -> Pipeline:
    return Pipeline([
        ("preprocessor", build_preprocessor(numeric_cols, categorical_cols)),
        ("model", build_model(name, task, y_train, params, n_jobs)),
    ])


def search_space(name: str, task: str, y_train) -> dict:
    """RandomizedSearchCV distributions, keyed by pipeline parameter name."""
    if name == "logistic_regression":
        return {
            "model__C": loguniform(1e-3, 1e2),
            "model__penalty": ["l1", "l2"],
            "model__class_weight": [None, "balanced"],
        }
    if name == "svm":
        return {
            "model__estimator__C": loguniform(0.1, 30),
            "model__estimator__gamma": loguniform(1e-3, 0.3),
            "model__estimator__class_weight": [None, "balanced"],
        }
    if name == "random_forest":
        # The card data is 3x larger, so its forest is kept smaller: a
        # 300-tree forest already takes ~5 minutes per fit on one core there.
        if task == "card":
            trees, depths = randint(100, 300), [8, 12, 16, 20]
        else:
            trees, depths = randint(100, 400), [10, 15, 20, 25, None]
        return {
            "model__n_estimators": trees,
            "model__max_depth": depths,
            "model__min_samples_leaf": randint(1, 10),
            "model__max_features": ["sqrt", "log2", 0.3],
            "model__class_weight": [None, "balanced"],
        }
    if name == "xgboost":
        ratio = positive_class_weight(y_train)
        weights = [1.0, ratio] if task == "hotel" else [1.0, float(np.sqrt(ratio)), ratio]
        return {
            "model__n_estimators": randint(200, 700),
            "model__max_depth": randint(3, 10),
            "model__learning_rate": loguniform(0.01, 0.3),
            "model__subsample": uniform(0.6, 0.4),
            "model__colsample_bytree": uniform(0.6, 0.4),
            "model__min_child_weight": randint(1, 10),
            "model__scale_pos_weight": [round(w, 3) for w in weights],
        }
    raise ValueError(f"Unknown model {name!r}")


def format_value(value) -> str:
    if value is None:
        return "None"
    if isinstance(value, (float, np.floating)):
        return f"{value:.4g}".replace(".", ",")
    return str(value)


def describe_distribution(dist) -> str:
    """Human-readable search range for the thesis table (decimal commas)."""
    if isinstance(dist, list):
        return "{" + "; ".join(format_value(v) for v in dist) + "}"
    name, args = dist.dist.name, dist.args
    if name == "randint":
        return f"cijeli broj iz [{args[0]}, {args[1] - 1}]"
    if name in ("loguniform", "reciprocal"):
        return f"log-uniformno iz [{format_value(args[0])}; {format_value(args[1])}]"
    if name == "uniform":
        return f"uniformno iz [{format_value(args[0])}; {format_value(args[0] + args[1])}]"
    return str(dist)


def _to_builtin(value):
    return value.item() if isinstance(value, np.generic) else value


def save_params(path: Path, params_by_model: dict) -> None:
    clean = {model: {k: _to_builtin(v) for k, v in params.items()}
             for model, params in params_by_model.items()}
    path.write_text(json.dumps(clean, indent=2, ensure_ascii=False), encoding="utf-8")


def load_params(path: Path) -> dict:
    """Tuned parameters per model, or {} if tuning.py has not been run."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def positive_scores(estimator, X) -> np.ndarray:
    """Score for the positive class.

    A probability where the model provides one, otherwise the SVM's decision
    value. ROC-AUC and PR-AUC depend only on the ordering of the scores, so
    both kinds are valid for them.
    """
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    return estimator.decision_function(X)


# Each metric receives (true labels, positive-class scores, hard predictions).
HOTEL_METRICS = {
    "roc_auc": lambda y, s, p: roc_auc_score(y, s),
    "accuracy": lambda y, s, p: accuracy_score(y, p),
    "precision": lambda y, s, p: precision_score(y, p, zero_division=0),
    "recall": lambda y, s, p: recall_score(y, p, zero_division=0),
    "f1": lambda y, s, p: f1_score(y, p, zero_division=0),
}
CARD_METRICS = {
    "pr_auc": lambda y, s, p: average_precision_score(y, s),
    "roc_auc": lambda y, s, p: roc_auc_score(y, s),
}


def repeated_cv(pipeline, X, y, metrics: dict, n_splits: int = 5, n_repeats: int = 2):
    """Repeated stratified k-fold CV on the training data.

    Returns per-fold metric values, plus the out-of-fold scores of the first
    repeat: every training row scored exactly once by a model that never saw
    it. Decision thresholds are chosen on these scores, never on the test set.
    """
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                 random_state=RANDOM_STATE)
    oof = np.full(len(y), np.nan)
    rows = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        fitted = clone(pipeline).fit(X.iloc[train_idx], y.iloc[train_idx])
        scores = positive_scores(fitted, X.iloc[val_idx])
        predictions = fitted.predict(X.iloc[val_idx])
        y_val = y.iloc[val_idx]
        rows.append({name: fn(y_val, scores, predictions) for name, fn in metrics.items()})
        if fold < n_splits:  # the first repeat covers every row once
            oof[val_idx] = scores
    return pd.DataFrame(rows), oof


def out_of_fold_scores(pipeline, X, y, n_splits: int = 5) -> np.ndarray:
    """The same out-of-fold scores repeated_cv returns (identical folds)."""
    _, oof = repeated_cv(pipeline, X, y, metrics={}, n_splits=n_splits, n_repeats=1)
    return oof


def summarize_folds(folds: pd.DataFrame) -> dict:
    summary = {}
    for metric in folds.columns:
        summary[f"cv_{metric}_mean"] = folds[metric].mean()
        summary[f"cv_{metric}_std"] = folds[metric].std()
    return summary


def pretty_feature_names(names, categorical_cols=()) -> list[str]:
    """'cat__country_PRT' -> 'country = PRT', 'num__lead_time' -> 'lead_time'."""
    # Longest names first, so 'deposit_type' is not mistaken for a prefix match
    # of a longer column that happens to start the same way.
    columns = sorted(categorical_cols, key=len, reverse=True)
    pretty = []
    for raw in names:
        name = raw.split("__", 1)[-1]
        if raw.startswith("cat__"):
            for column in columns:
                if name.startswith(column + "_"):
                    value = name[len(column) + 1:]
                    if value == "infrequent_sklearn":
                        value = "ostale (rijetke)"
                    name = f"{column} = {value}"
                    break
        pretty.append(name)
    return pretty
