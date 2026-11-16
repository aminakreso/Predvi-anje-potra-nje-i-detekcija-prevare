"""Supervised credit-card fraud detection on the ULB dataset.

Dataset: "Credit Card Fraud Detection" (Machine Learning Group, ULB) --
284,807 real card transactions from September 2013, of which 492 are
fraudulent (0.172%). Features V1-V28 are PCA-anonymized; Time and Amount
are the only raw columns. Target: Class (1 = fraud).

Because fraud is so rare, ACCURACY IS A USELESS METRIC here -- predicting
"never fraud" scores about 99.83%. Average precision (PR-AUC) is the primary
metric, and the decision threshold is tuned explicitly, since the default
0.5 cutoff is rarely the right operating point.

Protocol (shared with train.py, see modeling.py):
  - rows that are exact duplicates (identical model inputs and label) are
    removed, so no transaction is in both the training and the test split;
  - stratified 80/20 split; tuned hyperparameters from
    reports/card_best_params.json if tuning.py has been run;
  - repeated stratified 5-fold CV on the training split -> mean +/- std;
  - the decision threshold is chosen on OUT-OF-FOLD training predictions and
    then applied unchanged to the test split (choosing it on the test split
    would make the reported precision and recall optimistic);
  - the best model is chosen by mean CV PR-AUC.
Class imbalance is handled with class weights; imbalance.py compares that
choice with SMOTE oversampling.

Usage:
    python src/card_fraud.py
"""

import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

from modeling import (
    CARD_METRICS,
    MODEL_LABELS,
    MODEL_NAMES,
    build_pipeline,
    load_params,
    positive_scores,
    repeated_cv,
    split_data,
    summarize_folds,
)
from plotting import add_show_argument, confusion_matrix_figure, emit, num, plt, report_saved

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = ROOT / "data" / "raw" / "creditcard.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
PARAMS_PATH = REPORT_DIR / "card_best_params.json"
TARGET = "Class"
CLASS_LABELS = ["Legitimna", "Prevara"]

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


def load_modeling_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Features and target with duplicate rows removed."""
    df = engineer_features(load_data(path))
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Removed {n_before - len(df)} duplicate rows; {len(df)} remain, "
          f"{int(df[TARGET].sum())} of them fraudulent ({df[TARGET].mean() * 100:.3f}%).")
    return df.drop(columns=[TARGET]), df[TARGET]


def best_f1_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[float, float]:
    """Find the probability cutoff maximizing F1, instead of assuming 0.5."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # precision/recall have one more element than thresholds.
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    best_idx = int(np.argmax(f1))
    return float(thresholds[best_idx]), float(f1[best_idx])


def evaluate(estimator, X_test, y_test, threshold: float) -> dict:
    scores = positive_scores(estimator, X_test)
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "test_pr_auc": average_precision_score(y_test, scores),
        "test_roc_auc": roc_auc_score(y_test, scores),
        "threshold": threshold,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0,
        "test_accuracy": (tp + tn) / len(y_test),
        "frauds_caught": int(tp),
        "frauds_missed": int(fn),
        "false_alarms": int(fp),
        "true_negatives": int(tn),
    }


def plot_pr_curves(fitted: dict, X_test, y_test) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, pipeline in fitted.items():
        scores = positive_scores(pipeline, X_test)
        precision, recall, _ = precision_recall_curve(y_test, scores)
        ap = average_precision_score(y_test, scores)
        ax.plot(recall, precision, label=f"{MODEL_LABELS[name]} (AP = {num(ap)})")

    baseline = y_test.mean()
    ax.axhline(baseline, linestyle="--", color="gray",
               label=f"Slučajni model (= {num(baseline, 4)})")
    ax.set_xlabel("Odziv (udio otkrivenih prevara)")
    ax.set_ylabel("Preciznost (udio stvarnih prevara među uzbunama)")
    ax.set_title("Preciznost-odziv krivulje na testnom skupu – prevare")
    ax.legend(loc="lower left")
    emit(fig, "12_fraud_pr_curves.png")


def plot_threshold_tradeoff(pipeline, X_test, y_test, name: str, threshold: float) -> None:
    scores = positive_scores(pipeline, X_test)
    precision, recall, thresholds = precision_recall_curve(y_test, scores)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(thresholds, precision[:-1], label="Preciznost")
    ax.plot(thresholds, recall[:-1], label="Odziv")
    ax.axvline(threshold, color="black", linestyle=":",
               label=f"Odabrani prag ({num(threshold)}, izabran na treningu)")
    ax.set_xlabel("Prag odlučivanja")
    ax.set_ylabel("Vrijednost metrike")
    ax.set_title(f"Preciznost i odziv u zavisnosti od praga – {MODEL_LABELS[name]}")
    ax.legend(loc="lower left")
    emit(fig, "13_fraud_threshold_tradeoff.png")


def main(data_path: Path, n_repeats: int) -> None:
    print(f"Loading data from {data_path} ...")
    X, y = load_modeling_data(data_path)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Train: {len(X_train)} rows ({y_train.sum()} fraud, {y_train.mean() * 100:.3f}%)")
    print(f"Test:  {len(X_test)} rows ({y_test.sum()} fraud, {y_test.mean() * 100:.3f}%)")

    params = load_params(PARAMS_PATH)
    if params:
        print(f"Using tuned hyperparameters from {PARAMS_PATH.name}")
    else:
        print("No tuned hyperparameters found -- using defaults. Run tuning.py first "
              "for the thesis results.")

    rows, fitted = {}, {}

    # Chance level: a constant score gives PR-AUC equal to the fraud rate.
    baseline = DummyClassifier(strategy="prior")
    folds, _ = repeated_cv(baseline, X_train, y_train, CARD_METRICS, n_repeats=n_repeats)
    baseline_scores = positive_scores(baseline.fit(X_train, y_train), X_test)
    rows["baseline"] = {
        **summarize_folds(folds),
        "test_pr_auc": average_precision_score(y_test, baseline_scores),
        "test_roc_auc": roc_auc_score(y_test, baseline_scores),
    }

    for name in MODEL_NAMES:
        print(f"\n[{name}] repeated {n_repeats}x5-fold CV ...")
        pipeline = build_pipeline(name, "card", list(X.columns), [], y_train, params.get(name))
        folds, oof_scores = repeated_cv(pipeline, X_train, y_train, CARD_METRICS,
                                        n_repeats=n_repeats)
        summary = summarize_folds(folds)
        threshold, oof_f1 = best_f1_threshold(y_train.to_numpy(), oof_scores)
        print(f"[{name}] CV PR-AUC {summary['cv_pr_auc_mean']:.4f} "
              f"+/- {summary['cv_pr_auc_std']:.4f}; threshold {threshold:.4f} "
              f"(out-of-fold F1 {oof_f1:.4f})")

        pipeline.fit(X_train, y_train)
        fitted[name] = pipeline
        rows[name] = {**summary, "oof_f1": oof_f1, **evaluate(pipeline, X_test, y_test, threshold)}

    results = pd.DataFrame(rows).T
    results.insert(0, "model", [MODEL_LABELS[i] for i in results.index])
    best_name = results.loc[MODEL_NAMES, "cv_pr_auc_mean"].astype(float).idxmax()
    results["selected"] = results.index == best_name

    show_cols = ["cv_pr_auc_mean", "cv_pr_auc_std", "test_pr_auc", "test_roc_auc",
                 "test_precision", "test_recall", "frauds_caught", "frauds_missed",
                 "false_alarms"]
    print("\nResults (ranked by CV PR-AUC, the metric that matters here):")
    print(results[show_cols].astype(float).round(4).sort_values(
        "cv_pr_auc_mean", ascending=False).to_string())

    best_pipeline = fitted[best_name]
    best = rows[best_name]
    print(f"\nBest model (by mean CV PR-AUC): {best_name}")
    print(f"  At threshold {best['threshold']:.4f} (chosen on training data): "
          f"caught {best['frauds_caught']} / {best['frauds_caught'] + best['frauds_missed']} "
          f"frauds, missed {best['frauds_missed']}, with {best['false_alarms']} false alarms.")

    y_pred = (positive_scores(best_pipeline, X_test) >= best["threshold"]).astype(int)
    print("\n" + classification_report(y_test, y_pred, target_names=CLASS_LABELS, digits=4))
    print("Note: predicting 'never fraud' would score "
          f"{(1 - y_test.mean()) * 100:.2f}% accuracy and catch nothing.")

    plot_pr_curves(fitted, X_test, y_test)
    plot_threshold_tradeoff(best_pipeline, X_test, y_test, best_name, best["threshold"])
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    emit(confusion_matrix_figure(
        cm, CLASS_LABELS, f"Matrica konfuzije – {MODEL_LABELS[best_name]} (testni skup)"
    ), "29_card_confusion_matrix.png")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"pipeline": best_pipeline, "threshold": best["threshold"], "model_name": best_name},
        MODEL_DIR / "card_fraud_model.joblib",
        compress=3,
    )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(REPORT_DIR / "card_fraud_comparison.csv")
    print(f"\nSaved model + threshold to {MODEL_DIR / 'card_fraud_model.joblib'}")
    print(report_saved())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--n-repeats", type=int, default=2,
                        help="Repeats of stratified 5-fold CV (mean +/- std)")
    add_show_argument(parser)
    args = parser.parse_args()
    main(args.data_path, args.n_repeats)
