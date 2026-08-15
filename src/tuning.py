"""Hyperparameter search, repeated cross-validation, and trivial baselines.

Three things a thesis-level evaluation needs that a single train/test split
does not give you:

1. BASELINES -- a number is meaningless without something to beat. For the
   card-fraud data especially, a "predict the majority class" baseline scores
   99.83% accuracy, which is exactly why accuracy is the wrong metric.
2. TUNED HYPERPARAMETERS -- reported over a documented search space, so the
   result is reproducible rather than the product of hand-picked settings.
3. CONFIDENCE INTERVALS -- repeated stratified CV gives mean +/- std, so you
   can say whether model A really beats model B or the gap is noise.

Usage:
    python src/tuning.py --task hotel
    python src/tuning.py --task card
    python src/tuning.py --task both --n-iter 25
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import (
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.stats import loguniform, randint, uniform

from data_prep import TARGET, get_feature_lists, load_and_prepare

ROOT = Path(__file__).resolve().parents[1]
HOTEL_DATA = ROOT / "data" / "raw" / "hotel_bookings.csv"
CARD_DATA = ROOT / "data" / "raw" / "creditcard.csv"
REPORT_DIR = ROOT / "reports"
RANDOM_STATE = 42

warnings.filterwarnings("ignore")

# Search space is defined once, here, so the thesis can quote it verbatim.
XGB_SEARCH_SPACE = {
    "model__n_estimators": randint(200, 700),
    "model__max_depth": randint(3, 10),
    "model__learning_rate": loguniform(0.01, 0.3),
    "model__subsample": uniform(0.6, 0.4),
    "model__colsample_bytree": uniform(0.6, 0.4),
    "model__min_child_weight": randint(1, 10),
}


def evaluate_baselines(X, y, preprocessor, scoring: str, cv) -> pd.DataFrame:
    """Trivial reference points that any real model must beat."""
    rows = {}
    for name, strategy in [
        ("baseline_majority", "most_frequent"),
        ("baseline_stratified", "stratified"),
        ("baseline_prior", "prior"),
    ]:
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", DummyClassifier(strategy=strategy, random_state=RANDOM_STATE)),
        ])
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        rows[name] = {"mean": scores.mean(), "std": scores.std()}
    return pd.DataFrame(rows).T


def run_search(pipeline, space, X, y, scoring, n_iter, cv_folds):
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=space,
        n_iter=n_iter,
        scoring=scoring,
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    search.fit(X, y)
    return search


def repeated_cv(pipeline, X, y, scoring: str, n_repeats: int) -> tuple[float, float]:
    """Mean +/- std over repeated stratified CV -- the confidence interval."""
    cv = RepeatedStratifiedKFold(
        n_splits=5, n_repeats=n_repeats, random_state=RANDOM_STATE
    )
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return float(scores.mean()), float(scores.std())


def tune_hotel(n_iter: int, cv_folds: int, n_repeats: int) -> pd.DataFrame:
    from xgboost import XGBClassifier

    print("\n" + "=" * 70)
    print("HOTEL CANCELLATION -- tuning (scoring: roc_auc)")
    print("=" * 70)

    df = load_and_prepare(HOTEL_DATA)
    numeric_cols, categorical_cols = get_feature_lists(df)
    X, y = df[numeric_cols + categorical_cols], df[TARGET]
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ])

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    baselines = evaluate_baselines(X_train, y_train, preprocessor, "roc_auc", cv)
    print("\nBaselines (ROC-AUC):")
    print(baselines.round(4).to_string())

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", XGBClassifier(
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1
        )),
    ])

    print(f"\nRandomizedSearchCV: {n_iter} candidates x {cv_folds} folds ...")
    search = run_search(pipeline, XGB_SEARCH_SPACE, X_train, y_train,
                        "roc_auc", n_iter, cv_folds)

    print(f"\nBest CV ROC-AUC: {search.best_score_:.4f}")
    print("Best params:")
    for key, value in sorted(search.best_params_.items()):
        print(f"  {key.replace('model__', '')}: {value}")

    mean, std = repeated_cv(search.best_estimator_, X_train, y_train, "roc_auc", n_repeats)
    print(f"\nRepeated CV (5-fold x {n_repeats}): ROC-AUC {mean:.4f} +/- {std:.4f}")

    rows = baselines.copy()
    rows.loc["xgboost_tuned"] = {"mean": mean, "std": std}
    rows.to_csv(REPORT_DIR / "hotel_tuning_results.csv")

    pd.Series(search.best_params_).to_csv(REPORT_DIR / "hotel_best_params.csv")
    return rows


def tune_card(n_iter: int, cv_folds: int, n_repeats: int) -> pd.DataFrame:
    from xgboost import XGBClassifier

    print("\n" + "=" * 70)
    print("CARD FRAUD -- tuning (scoring: average_precision / PR-AUC)")
    print("=" * 70)

    if not CARD_DATA.exists():
        print(f"Skipping -- {CARD_DATA} not found.")
        return pd.DataFrame()

    from card_fraud import engineer_features, load_data

    df = engineer_features(load_data(CARD_DATA))
    X, y = df.drop(columns=["Class"]), df["Class"]
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # No categorical columns here -- V1-V28 are already numeric PCA components.
    preprocessor = ColumnTransformer([("num", StandardScaler(), list(X.columns))])
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    baselines = evaluate_baselines(X_train, y_train, preprocessor, "average_precision", cv)
    print("\nBaselines (PR-AUC):")
    print(baselines.round(5).to_string())
    print(f"  (chance level = fraud prevalence = {y_train.mean():.5f})")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", XGBClassifier(
            eval_metric="aucpr",
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

    print(f"\nRandomizedSearchCV: {n_iter} candidates x {cv_folds} folds ...")
    search = run_search(pipeline, XGB_SEARCH_SPACE, X_train, y_train,
                        "average_precision", n_iter, cv_folds)

    print(f"\nBest CV PR-AUC: {search.best_score_:.4f}")
    print("Best params:")
    for key, value in sorted(search.best_params_.items()):
        print(f"  {key.replace('model__', '')}: {value}")

    mean, std = repeated_cv(search.best_estimator_, X_train, y_train,
                            "average_precision", n_repeats)
    print(f"\nRepeated CV (5-fold x {n_repeats}): PR-AUC {mean:.4f} +/- {std:.4f}")
    print("  This +/- matters: with only ~394 training frauds, fold-to-fold "
          "variance is large and single-split numbers can mislead.")

    rows = baselines.copy()
    rows.loc["xgboost_tuned"] = {"mean": mean, "std": std}
    rows.to_csv(REPORT_DIR / "card_tuning_results.csv")

    pd.Series(search.best_params_).to_csv(REPORT_DIR / "card_best_params.csv")
    return rows


def main(task: str, n_iter: int, cv_folds: int, n_repeats: int) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if task in ("hotel", "both"):
        tune_hotel(n_iter, cv_folds, n_repeats)
    if task in ("card", "both"):
        tune_card(n_iter, cv_folds, n_repeats)

    print(f"\nSaved tuning results + best params to {REPORT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["hotel", "card", "both"], default="both")
    parser.add_argument("--n-iter", type=int, default=15,
                        help="RandomizedSearchCV candidates (higher = slower, better)")
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--n-repeats", type=int, default=2,
                        help="Repeats for the confidence-interval CV")
    args = parser.parse_args()
    main(args.task, args.n_iter, args.cv_folds, args.n_repeats)
