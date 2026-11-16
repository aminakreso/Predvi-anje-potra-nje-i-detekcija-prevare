"""Class weights vs. SMOTE for the card-fraud models.

Compares three ways of dealing with 0.17% fraud, with stratified 5-fold CV
on the TRAINING split only (choosing the imbalance strategy is part of
building the model, so the test split stays untouched):

  - no correction;
  - class weights (class_weight="balanced" / XGBoost scale_pos_weight) --
    the strategy card_fraud.py uses;
  - SMOTE oversampling inside the pipeline: synthetic frauds are generated
    from each training fold only, never from the rows the fold is scored on.

Each model starts from its tuned hyperparameters (if tuning.py has been run)
with only the weighting parameter changed per strategy.

The SVM is not part of this comparison: it already trains on an undersampled
subset (modeling.SubsampledClassifier), which is itself an imbalance
strategy, and SMOTE would multiply its training set far beyond what a kernel
SVM can handle.

Usage:
    python src/imbalance.py
"""

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline

from card_fraud import load_modeling_data
from modeling import (
    CARD_METRICS,
    MODEL_LABELS,
    RANDOM_STATE,
    build_model,
    build_preprocessor,
    load_params,
    positive_class_weight,
    repeated_cv,
    split_data,
    summarize_folds,
)
from plotting import add_show_argument, emit, num, plt, report_saved

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = ROOT / "data" / "raw" / "creditcard.csv"
REPORT_DIR = ROOT / "reports"
PARAMS_PATH = REPORT_DIR / "card_best_params.json"

MODELS = ["logistic_regression", "random_forest", "xgboost"]
STRATEGIES = {
    "none": "Bez korekcije",
    "class_weights": "Težine klasa",
    "smote": "SMOTE",
}

warnings.filterwarnings("ignore")


def weighting(name: str, strategy: str, y_train) -> dict:
    """The one parameter that differs between strategies."""
    if name == "xgboost":
        return {"scale_pos_weight": positive_class_weight(y_train) if strategy == "class_weights" else 1.0}
    return {"class_weight": "balanced" if strategy == "class_weights" else None}


def build(name: str, strategy: str, columns, y_train, tuned: dict):
    params = {k: v for k, v in tuned.items() if k not in ("class_weight", "scale_pos_weight")}
    params.update(weighting(name, strategy, y_train))
    model = build_model(name, "card", y_train, params)
    preprocessor = build_preprocessor(columns)
    if strategy == "smote":
        return ImbPipeline([("preprocessor", preprocessor),
                            ("smote", SMOTE(random_state=RANDOM_STATE)),
                            ("model", model)])
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def plot_comparison(results: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.25
    x = np.arange(len(MODELS))
    for i, (strategy, label) in enumerate(STRATEGIES.items()):
        subset = results[results["strategy"] == strategy].set_index("model_key").loc[MODELS]
        bars = ax.bar(x + (i - 1) * width, subset["cv_pr_auc_mean"], width,
                      yerr=subset["cv_pr_auc_std"], capsize=3, label=label)
        for bar, value in zip(bars, subset["cv_pr_auc_mean"]):
            ax.annotate(num(value), (bar.get_x() + bar.get_width() / 2, value),
                        ha="center", va="bottom", fontsize=7, xytext=(0, 3),
                        textcoords="offset points")
    ax.set_xticks(x, [MODEL_LABELS[m] for m in MODELS])
    ax.set_ylabel("PR-AUC (5-struka unakrsna validacija, ± std)")
    ax.set_ylim(0, 1)
    ax.set_title("Poređenje postupaka za nebalansirane klase – prevare na karticama")
    ax.legend(loc="lower right")
    emit(fig, "33_card_imbalance_comparison.png")


def main(data_path: Path) -> None:
    X, y = load_modeling_data(data_path)
    X_train, _, y_train, _ = split_data(X, y)
    tuned = load_params(PARAMS_PATH)
    columns = list(X.columns)

    rows = []
    for name in MODELS:
        for strategy, label in STRATEGIES.items():
            start = time.time()
            pipeline = build(name, strategy, columns, y_train, tuned.get(name, {}))
            folds, _ = repeated_cv(pipeline, X_train, y_train, CARD_METRICS, n_repeats=1)
            summary = summarize_folds(folds)
            print(f"[{name:<20}] {label:<14} PR-AUC {summary['cv_pr_auc_mean']:.4f} "
                  f"+/- {summary['cv_pr_auc_std']:.4f}  ({time.time() - start:.0f}s)")
            rows.append({"model_key": name, "model": MODEL_LABELS[name],
                         "strategy": strategy, "strategy_label": label, **summary})

    results = pd.DataFrame(rows)
    plot_comparison(results)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(REPORT_DIR / "card_imbalance_comparison.csv", index=False)
    print(f"\nSaved {REPORT_DIR / 'card_imbalance_comparison.csv'}")
    print(report_saved())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    add_show_argument(parser)
    args = parser.parse_args()
    main(args.data_path)
