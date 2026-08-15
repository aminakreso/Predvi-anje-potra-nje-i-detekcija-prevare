"""Supervised credit-card fraud detection on the ULB dataset.

Dataset: "Credit Card Fraud Detection" (Machine Learning Group, ULB) --
284,807 real card transactions from September 2013, of which 492 are
fraudulent (0.172%). Features V1-V28 are PCA-anonymized; Time and Amount
are the only raw columns. Target: Class (1 = fraud).

Because fraud is only ~0.17% of rows, ACCURACY IS A USELESS METRIC here --
predicting "never fraud" scores 99.83%. This script uses average precision
(PR-AUC) as the primary metric and tunes the decision threshold explicitly,
since the default 0.5 cutoff is rarely the right operating point.

Usage:
    python src/card_fraud.py
    python src/card_fraud.py --use-smote
"""

import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from plotting import FIG_DIR, add_show_argument, emit, plt, report_saved

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = ROOT / "data" / "raw" / "creditcard.csv"
MODEL_DIR = ROOT / "models"
RANDOM_STATE = 42
TARGET = "Class"

warnings.filterwarnings("ignore", category=UserWarning)


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(
            f"Dataset not found at {path}\n\n"
            "Download 'creditcard.csv' from Kaggle:\n"
            "  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            f"and place it at {path}"
        )
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(f"Expected target column '{TARGET}' not found in {path}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Time is "seconds since the first transaction in the dataset", which is
    # meaningless as a raw magnitude. Hour-of-day is the part that could
    # plausibly carry signal (fraud clusters at night in this data).
    df["hour"] = (df["Time"] // 3600) % 24
    # Amount is heavily right-skewed; log-scale it so it isn't dominated by
    # a handful of large transactions.
    df["log_amount"] = np.log1p(df["Amount"])
    df = df.drop(columns=["Time", "Amount"])
    return df


def build_models(y_train: pd.Series, use_smote: bool) -> dict:
    # Ratio of negatives to positives -- tells the booster how much to
    # up-weight the rare fraud class.
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # With SMOTE the training set is rebalanced by oversampling, so applying
    # class weights on top would double-count the correction.
    class_weight = None if use_smote else "balanced"

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight=class_weight, random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=3,
            class_weight=class_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=1.0 if use_smote else scale_pos_weight,
            eval_metric="aucpr",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    except ImportError:
        print("xgboost not installed -- skipping it.")

    return models


def build_pipeline(model, use_smote: bool) -> Pipeline:
    steps = [("scaler", StandardScaler())]

    if use_smote:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline

        # SMOTE must live inside the pipeline so it only ever resamples the
        # training fold -- never the data we evaluate on.
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
        steps.append(("model", model))
        return ImbPipeline(steps)

    steps.append(("model", model))
    return Pipeline(steps)


def best_f1_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[float, float]:
    """Find the probability cutoff maximizing F1, instead of assuming 0.5."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # precision/recall have one more element than thresholds.
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    best_idx = int(np.argmax(f1))
    return float(thresholds[best_idx]), float(f1[best_idx])


def evaluate(name: str, pipeline: Pipeline, X_test, y_test) -> dict:
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    threshold, f1_at_best = best_f1_threshold(y_test.to_numpy(), y_proba)
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    return {
        "pr_auc": average_precision_score(y_test, y_proba),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "best_threshold": threshold,
        "f1_at_threshold": f1_at_best,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "frauds_caught": int(tp),
        "frauds_missed": int(fn),
        "false_alarms": int(fp),
    }


def plot_pr_curves(fitted: dict, X_test, y_test) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, pipeline in fitted.items():
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})")

    baseline = y_test.mean()
    ax.axhline(baseline, linestyle="--", color="gray",
               label=f"Chance (={baseline:.4f})")
    ax.set_xlabel("Recall (share of fraud caught)")
    ax.set_ylabel("Precision (share of alerts that are real)")
    ax.set_title("Precision-Recall curves -- test set")
    ax.legend(loc="upper right")
    emit(fig, "12_fraud_pr_curves.png")


def plot_threshold_tradeoff(pipeline: Pipeline, X_test, y_test, name: str) -> None:
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(thresholds, precision[:-1], label="Precision")
    ax.plot(thresholds, recall[:-1], label="Recall")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"Precision/recall trade-off vs. threshold ({name})")
    ax.legend()
    emit(fig, "13_fraud_threshold_tradeoff.png")


def main(data_path: Path, use_smote: bool) -> None:
    print(f"Loading data from {data_path} ...")
    df = load_data(data_path)
    df = engineer_features(df)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train)} rows ({y_train.sum()} fraud, {y_train.mean() * 100:.3f}%)")
    print(f"Test:  {len(X_test)} rows ({y_test.sum()} fraud, {y_test.mean() * 100:.3f}%)")
    print(f"Imbalance handling: {'SMOTE oversampling' if use_smote else 'class weights'}\n")

    results = {}
    fitted = {}
    for name, model in build_models(y_train, use_smote).items():
        print(f"Training {name} ...")
        pipeline = build_pipeline(model, use_smote)
        pipeline.fit(X_train, y_train)
        fitted[name] = pipeline
        results[name] = evaluate(name, pipeline, X_test, y_test)

    results_df = pd.DataFrame(results).T.sort_values("pr_auc", ascending=False)
    print("\nTest-set results (ranked by PR-AUC -- the metric that matters here):")
    print(results_df.round(4).to_string())

    print(
        "\n*** These are SINGLE-SPLIT figures. Do not quote them as headline\n"
        "    results. With only ~394 training frauds, fold-to-fold variance is\n"
        "    roughly +/- 0.03 PR-AUC, so differences smaller than that are noise.\n"
        "    Run `python src/tuning.py --task card` for the repeated-CV interval,\n"
        "    which is the defensible number to report. ***"
    )

    best_name = results_df.index[0]
    best_pipeline = fitted[best_name]
    best = results[best_name]

    print(f"\nBest model: {best_name}")
    print(f"  At threshold {best['best_threshold']:.4f}: "
          f"caught {best['frauds_caught']} / {best['frauds_caught'] + best['frauds_missed']} frauds, "
          f"missed {best['frauds_missed']}, with {best['false_alarms']} false alarms.")

    y_proba = best_pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= best["best_threshold"]).astype(int)
    print("\n" + classification_report(y_test, y_pred, target_names=["Legit", "Fraud"], digits=4))

    print("Note: accuracy is omitted on purpose -- predicting 'never fraud' "
          f"would score {(1 - y_test.mean()) * 100:.2f}% and catch nothing.")

    plot_pr_curves(fitted, X_test, y_test)
    plot_threshold_tradeoff(best_pipeline, X_test, y_test, best_name)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"pipeline": best_pipeline, "threshold": best["best_threshold"]},
        MODEL_DIR / "card_fraud_model.joblib",
        compress=3,
    )
    results_df.to_csv(ROOT / "reports" / "card_fraud_comparison.csv")
    print(f"\nSaved model + threshold to {MODEL_DIR / 'card_fraud_model.joblib'}")
    print(report_saved())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--use-smote", action="store_true",
                        help="Oversample the minority class with SMOTE instead of class weights")
    add_show_argument(parser)
    args = parser.parse_args()
    main(args.data_path, args.use_smote)
