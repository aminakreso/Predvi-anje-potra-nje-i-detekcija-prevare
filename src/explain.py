"""Model explainability (SHAP) and cost-sensitive threshold selection.

Two additions that move the project from "the model scores X" to "here is
WHY it decides, and here is the operating point that actually minimises
money lost":

1. SHAP -- global feature attribution plus per-prediction explanations.
   Unlike the built-in feature_importances_, SHAP values are signed and
   per-instance, so you can answer "why was THIS booking flagged?".

2. COST-SENSITIVE THRESHOLDS -- the F1-optimal cutoff used in card_fraud.py
   implicitly assumes a false alarm and a missed fraud are equally bad. They
   are not. Here the cost of a missed fraud is the ACTUAL transaction amount
   (real money lost) and a false alarm costs a fixed manual-review fee.
   Both thresholds are chosen on out-of-fold TRAINING predictions and only
   then evaluated on the test split; the test-set optimum is reported purely
   as a reference for how much was left on the table.

Usage:
    python src/explain.py --task card
    python src/explain.py --task hotel
    python src/explain.py --task both --review-cost 3.0
"""

import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from modeling import (
    MODEL_LABELS,
    RANDOM_STATE,
    out_of_fold_scores,
    positive_scores,
    pretty_feature_names,
    split_data,
)
from plotting import add_show_argument, emit, num, plt, report_saved

ROOT = Path(__file__).resolve().parents[1]
HOTEL_DATA = ROOT / "data" / "raw" / "hotel_bookings.csv"
CARD_DATA = ROOT / "data" / "raw" / "creditcard.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
# SHAP cost grows with the number of rows explained and with tree size. A
# sample is plenty for a global summary; deep random forests get a smaller one
# because their trees have ~10x more leaves than the boosted trees.
SHAP_SAMPLE = 2000
SHAP_SAMPLE_FOREST = 500
DEFAULT_GRID = np.unique(np.round(np.linspace(0.001, 0.999, 400), 4))

warnings.filterwarnings("ignore")


def shap_sample(pipeline, X_test: pd.DataFrame) -> pd.DataFrame:
    is_forest = type(pipeline.named_steps["model"]).__name__ == "RandomForestClassifier"
    n = SHAP_SAMPLE_FOREST if is_forest else SHAP_SAMPLE
    return X_test.sample(min(n, len(X_test)), random_state=RANDOM_STATE)


def shap_summary(pipeline, X_sample: pd.DataFrame, categorical_cols, prefix: str,
                 title: str) -> pd.Series:
    """TreeExplainer over the pipeline's transformed feature space."""
    model = pipeline.named_steps["model"]
    preprocessing = pipeline[:-1]

    X_transformed = preprocessing.transform(X_sample)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    feature_names = pretty_feature_names(preprocessing.get_feature_names_out(), categorical_cols)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)
    # Depending on the model and SHAP version, binary classifiers return one
    # array per class (a list, or a 3-D array) -- keep the positive class.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    shap.summary_plot(
        shap_values, X_transformed, feature_names=feature_names, show=False,
        max_display=20, plot_size=(10, 8), color_bar_label="Vrijednost obilježja",
    )
    fig = plt.gcf()
    fig.axes[0].set_xlabel("SHAP vrijednost (doprinos predikciji pozitivne klase)")
    colorbar = fig.axes[-1]
    colorbar.set_yticks(colorbar.get_yticks(), ["Niska", "Visoka"])
    fig.axes[0].set_title(title, fontsize=11)
    emit(fig, f"{prefix}_shap_beeswarm.png")

    shap.summary_plot(
        shap_values, X_transformed, feature_names=feature_names, plot_type="bar",
        show=False, max_display=20, plot_size=(10, 8),
    )
    fig = plt.gcf()
    fig.axes[0].set_xlabel("Prosječna apsolutna SHAP vrijednost")
    fig.axes[0].set_title(f"{title} – prosječni uticaj", fontsize=11)
    emit(fig, f"{prefix}_shap_importance.png")

    return pd.Series(
        np.abs(shap_values).mean(axis=0), index=feature_names
    ).sort_values(ascending=False)


def cost_at_thresholds(y_true, scores, amounts, review_cost: float, thresholds) -> pd.DataFrame:
    """Exact cost of flagging every transaction with score >= threshold.

    A missed fraud (false negative) costs the transaction amount -- the money
    that actually walks out the door. A false alarm costs `review_cost`, the
    price of a human looking at a legitimate transaction. Computed with one
    sort and cumulative sums, so any number of thresholds is cheap.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    amounts = np.asarray(amounts, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)

    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    is_fraud = y_true[order] == 1
    # Prefix sums with a leading 0: entry k describes the k highest-scored rows.
    caught_amount = np.concatenate([[0.0], np.cumsum(np.where(is_fraud, amounts[order], 0.0))])
    caught_count = np.concatenate([[0], np.cumsum(is_fraud)])
    alarm_count = np.concatenate([[0], np.cumsum(~is_fraud)])

    # Number of rows with score >= threshold (scores are sorted descending).
    n_flagged = np.searchsorted(-sorted_scores, -thresholds, side="right")
    fraud_loss = caught_amount[-1] - caught_amount[n_flagged]
    false_alarms = alarm_count[n_flagged]
    review_spend = false_alarms * review_cost
    return pd.DataFrame({
        "threshold": thresholds,
        "fraud_loss": fraud_loss,
        "review_cost": review_spend,
        "total_cost": fraud_loss + review_spend,
        "frauds_missed": caught_count[-1] - caught_count[n_flagged],
        "false_alarms": false_alarms,
    })


def cost_curve(y_true, y_proba, amounts, review_cost: float, thresholds=None) -> pd.DataFrame:
    """Total cost over a threshold grid (default: 400 points in (0, 1))."""
    grid = DEFAULT_GRID if thresholds is None else thresholds
    return cost_at_thresholds(y_true, y_proba, amounts, review_cost, grid)


def optimal_cost_threshold(y_true, scores, amounts, review_cost: float) -> float:
    """Threshold with the lowest total cost, searched over every distinct score."""
    candidates = np.unique(scores)
    costs = cost_at_thresholds(y_true, scores, amounts, review_cost, candidates)
    return float(costs.loc[costs["total_cost"].idxmin(), "threshold"])


def plot_cost_curve(costs: pd.DataFrame, cost_threshold: float, f1_threshold: float) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(costs["threshold"], costs["total_cost"], label="Ukupni trošak", color="#C44E52")
    ax.plot(costs["threshold"], costs["fraud_loss"], label="Gubitak zbog propuštenih prevara",
            alpha=0.7, linestyle="--")
    ax.plot(costs["threshold"], costs["review_cost"], label="Trošak ručnog pregleda",
            alpha=0.7, linestyle="--")
    ax.axvline(cost_threshold, color="green",
               label=f"Troškovno optimalan prag ({num(cost_threshold, 4)})")
    ax.axvline(f1_threshold, color="gray", linestyle=":",
               label=f"F1-optimalan prag ({num(f1_threshold, 4)})")
    ax.set_xlabel("Prag odlučivanja")
    ax.set_ylabel("Trošak (EUR)")
    ax.set_title("Trošak na testnom skupu u zavisnosti od praga (pragovi izabrani na treningu)")
    ax.legend(fontsize=8)
    emit(fig, "20_fraud_cost_curve.png")


def explain_card(review_cost: float) -> None:
    print("\n" + "=" * 70)
    print("CARD FRAUD -- cost-sensitive thresholds + SHAP")
    print("=" * 70)

    model_path = MODEL_DIR / "card_fraud_model.joblib"
    if not model_path.exists() or not CARD_DATA.exists():
        print("Skipping -- run card_fraud.py first (model or dataset missing).")
        return

    from card_fraud import load_modeling_data

    bundle = joblib.load(model_path)
    pipeline, f1_threshold = bundle["pipeline"], bundle["threshold"]
    model_name = bundle.get("model_name", "xgboost")

    X, y = load_modeling_data(CARD_DATA)
    X_train, X_test, y_train, y_test = split_data(X, y)  # same split as card_fraud.py
    amounts_train = np.expm1(X_train["log_amount"].to_numpy())  # undo the log1p
    amounts_test = np.expm1(X_test["log_amount"].to_numpy())

    print("Computing out-of-fold training scores (same folds as card_fraud.py) ...")
    oof = out_of_fold_scores(pipeline, X_train, y_train)
    cost_threshold = optimal_cost_threshold(y_train.to_numpy(), oof, amounts_train, review_cost)

    test_scores = positive_scores(pipeline, X_test)
    y_test_np = y_test.to_numpy()
    oracle_threshold = optimal_cost_threshold(y_test_np, test_scores, amounts_test, review_cost)

    strategies = {
        "Bez modela (nijedna transakcija se ne zaustavlja)": np.inf,
        "F1-optimalan prag (izabran na treningu)": f1_threshold,
        "Troškovno optimalan prag (izabran na treningu)": cost_threshold,
        "Optimum na testnom skupu (samo referenca, nedostupan unaprijed)": oracle_threshold,
    }
    summary = cost_at_thresholds(y_test_np, test_scores, amounts_test, review_cost,
                                 list(strategies.values()))
    summary.insert(0, "strategy", list(strategies))
    do_nothing = summary.loc[0, "total_cost"]
    summary["reduction_vs_no_model_%"] = (1 - summary["total_cost"] / do_nothing) * 100
    summary["review_cost_per_alarm"] = review_cost
    summary["model"] = MODEL_LABELS.get(model_name, model_name)

    print(f"\nModel: {model_name}; manual review cost: EUR {review_cost:.2f} per false alarm")
    print(summary.drop(columns=["model", "review_cost_per_alarm"]).round(4).to_string(index=False))

    costs = cost_curve(y_test_np, test_scores, amounts_test, review_cost)
    plot_cost_curve(costs, cost_threshold, f1_threshold)
    costs.to_csv(REPORT_DIR / "card_fraud_cost_curve.csv", index=False)
    summary.to_csv(REPORT_DIR / "card_cost_summary.csv", index=False)

    print("\nComputing SHAP values ...")
    sample = shap_sample(pipeline, X_test)
    try:
        importance = shap_summary(pipeline, sample, [], "19_card",
                                  "Prevare na karticama – SHAP atribucija obilježja")
        print("\nTop 10 features by mean |SHAP|:")
        print(importance.head(10).round(4).to_string())
        importance.to_csv(REPORT_DIR / "card_shap_importance.csv")
    except Exception as exc:  # linear models and the SVM have no TreeExplainer
        print(f"SHAP skipped: {exc}")


def explain_hotel() -> None:
    print("\n" + "=" * 70)
    print("HOTEL CANCELLATION -- SHAP")
    print("=" * 70)

    model_path = MODEL_DIR / "best_model.joblib"
    if not model_path.exists():
        print("Skipping -- run train.py first (model missing).")
        return

    from data_prep import load_modeling_data

    pipeline = joblib.load(model_path)
    X, y, _, categorical_cols = load_modeling_data(HOTEL_DATA)
    _, X_test, _, _ = split_data(X, y)  # same split as train.py

    print("Computing SHAP values ...")
    sample = shap_sample(pipeline, X_test)
    try:
        importance = shap_summary(pipeline, sample, categorical_cols, "18_hotel",
                                  "Otkazivanje rezervacija – SHAP atribucija obilježja")
        print("\nTop 10 features by mean |SHAP|:")
        print(importance.head(10).round(4).to_string())
        importance.to_csv(REPORT_DIR / "hotel_shap_importance.csv")
    except Exception as exc:
        print(f"SHAP skipped: {exc}")


def main(task: str, review_cost: float) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if task in ("hotel", "both"):
        explain_hotel()
    if task in ("card", "both"):
        explain_card(review_cost)
    print("\n" + report_saved())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["hotel", "card", "both"], default="both")
    parser.add_argument("--review-cost", type=float, default=3.0,
                        help="EUR cost of manually reviewing one flagged transaction")
    add_show_argument(parser)
    args = parser.parse_args()
    main(args.task, args.review_cost)
