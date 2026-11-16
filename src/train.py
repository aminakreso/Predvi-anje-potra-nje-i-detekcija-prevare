"""Train and compare the four cancellation-prediction models (hotel bookings).

Protocol (shared with card_fraud.py, see modeling.py):
  - duplicate rows removed, stratified 80/20 train/test split;
  - tuned hyperparameters from reports/hotel_best_params.json (written by
    tuning.py), otherwise the defaults in modeling.py;
  - repeated stratified 5-fold CV on the training split -> mean +/- std;
  - the best model is chosen by mean CV ROC-AUC -- never by the test set --
    and the test split is used once, for the final evaluation.

Usage:
    python src/train.py
    python src/train.py --n-repeats 3
"""

import argparse
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from data_prep import load_modeling_data
from modeling import (
    HOTEL_METRICS,
    MODEL_LABELS,
    MODEL_NAMES,
    build_pipeline,
    load_params,
    positive_scores,
    pretty_feature_names,
    repeated_cv,
    split_data,
    summarize_folds,
)
from plotting import add_show_argument, confusion_matrix_figure, emit, num, plt, report_saved

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = ROOT / "data" / "raw" / "hotel_bookings.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
PARAMS_PATH = REPORT_DIR / "hotel_best_params.json"
CLASS_LABELS = ["Nije otkazana", "Otkazana"]

warnings.filterwarnings("ignore", category=UserWarning)


def evaluate(estimator, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    scores = positive_scores(estimator, X_test)
    y_pred = estimator.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    return {
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred, zero_division=0),
        "test_recall": recall_score(y_test, y_pred, zero_division=0),
        "test_f1": f1_score(y_test, y_pred, zero_division=0),
        "test_roc_auc": roc_auc_score(y_test, scores),
        "test_pr_auc": average_precision_score(y_test, scores),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def plot_roc_curves(fitted: dict, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for name, pipeline in fitted.items():
        scores = positive_scores(pipeline, X_test)
        fpr, tpr, _ = roc_curve(y_test, scores)
        auc = roc_auc_score(y_test, scores)
        ax.plot(fpr, tpr, label=f"{MODEL_LABELS[name]} (AUC = {num(auc)})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Slučajno pogađanje (AUC = 0,5)")
    ax.set_xlabel("Stopa lažno pozitivnih (FPR)")
    ax.set_ylabel("Stopa stvarno pozitivnih (TPR, odziv)")
    ax.set_title("ROC krivulje na testnom skupu – otkazivanje rezervacija")
    ax.legend(loc="lower right")
    emit(fig, "07_roc_curves.png")


def plot_feature_importance(pipeline, model_name: str, categorical_cols, top_n: int = 20) -> None:
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        print(f"{model_name} has no built-in feature importances -- see explain.py (SHAP).")
        return

    raw_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    names = pretty_feature_names(raw_names, categorical_cols)
    importances = pd.Series(model.feature_importances_, index=names)
    top = importances.sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(8, 8))
    top.sort_values().plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_xlabel("Relativna važnost obilježja")
    ax.set_title(f"{top_n} najvažnijih obilježja – {MODEL_LABELS[model_name]}")
    emit(fig, "08_feature_importance.png")


def main(data_path: Path, n_repeats: int) -> None:
    print(f"Loading data from {data_path} ...")
    X, y, numeric_cols, categorical_cols = load_modeling_data(data_path)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Train rows: {len(X_train)}  Test rows: {len(X_test)}  "
          f"Cancellation rate (train): {y_train.mean():.3f}")

    params = load_params(PARAMS_PATH)
    if params:
        print(f"Using tuned hyperparameters from {PARAMS_PATH.name}")
    else:
        print("No tuned hyperparameters found -- using defaults. Run tuning.py first "
              "for the thesis results.")

    rows, fitted = {}, {}

    # Trivial reference point: always predicts the majority class.
    baseline = DummyClassifier(strategy="prior")
    folds, _ = repeated_cv(baseline, X_train, y_train, HOTEL_METRICS, n_repeats=n_repeats)
    rows["baseline"] = {**summarize_folds(folds), **evaluate(baseline.fit(X_train, y_train), X_test, y_test)}

    for name in MODEL_NAMES:
        print(f"\n[{name}] repeated {n_repeats}x5-fold CV ...")
        pipeline = build_pipeline(name, "hotel", numeric_cols, categorical_cols, y_train,
                                  params.get(name))
        folds, _ = repeated_cv(pipeline, X_train, y_train, HOTEL_METRICS, n_repeats=n_repeats)
        summary = summarize_folds(folds)
        print(f"[{name}] CV ROC-AUC {summary['cv_roc_auc_mean']:.4f} "
              f"+/- {summary['cv_roc_auc_std']:.4f}")

        pipeline.fit(X_train, y_train)
        fitted[name] = pipeline
        rows[name] = {**summary, **evaluate(pipeline, X_test, y_test)}

    results = pd.DataFrame(rows).T
    results.insert(0, "model", [MODEL_LABELS[i] for i in results.index])

    # Model selection on the training data only; the test set just reports.
    best_name = results.loc[MODEL_NAMES, "cv_roc_auc_mean"].astype(float).idxmax()
    results["selected"] = results.index == best_name

    show_cols = ["cv_roc_auc_mean", "cv_roc_auc_std", "test_accuracy", "test_precision",
                 "test_recall", "test_f1", "test_roc_auc"]
    print("\nResults (CV on the training split, then the test split):")
    print(results[show_cols].astype(float).round(4).to_string())

    best_pipeline = fitted[best_name]
    y_pred = best_pipeline.predict(X_test)
    print(f"\nBest model (by mean CV ROC-AUC): {best_name}")
    print(classification_report(y_test, y_pred, target_names=CLASS_LABELS, digits=4))

    plot_roc_curves(fitted, X_test, y_test)
    plot_feature_importance(best_pipeline, best_name, categorical_cols)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    emit(confusion_matrix_figure(
        cm, CLASS_LABELS, f"Matrica konfuzije – {MODEL_LABELS[best_name]} (testni skup)"
    ), "09_confusion_matrix.png")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_DIR / "best_model.joblib", compress=3)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(REPORT_DIR / "model_comparison.csv")
    print(f"\nSaved best model to {MODEL_DIR / 'best_model.joblib'}")
    print(report_saved())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--n-repeats", type=int, default=2,
                        help="Repeats of stratified 5-fold CV (mean +/- std)")
    add_show_argument(parser)
    args = parser.parse_args()
    main(args.data_path, args.n_repeats)
