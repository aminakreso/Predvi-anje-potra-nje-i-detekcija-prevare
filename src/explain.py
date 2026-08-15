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
   (real money lost) and a false alarm costs a fixed manual-review fee, so
   the optimal threshold is derived from economics rather than assumed.

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
from sklearn.model_selection import train_test_split

from plotting import FIG_DIR, add_show_argument, emit, plt, report_saved

ROOT = Path(__file__).resolve().parents[1]
HOTEL_DATA = ROOT / "data" / "raw" / "hotel_bookings.csv"
CARD_DATA = ROOT / "data" / "raw" / "creditcard.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
RANDOM_STATE = 42
SHAP_SAMPLE = 2000  # SHAP is O(n); a sample is plenty for a global summary

warnings.filterwarnings("ignore")


def save_fig(name: str) -> None:
    # SHAP draws onto the current figure, so let emit() pick it up via gcf().
    emit(None, name)


def shap_summary(pipeline, X_sample: pd.DataFrame, prefix: str, title: str) -> pd.Series:
    """TreeExplainer over the pipeline's transformed feature space.

    Pipelines in this project name their preprocessing step differently
    (`preprocessor` for the hotel model, `scaler` for the card model), so
    everything up to the final estimator is applied generically rather than
    looked up by name.
    """
    model = pipeline.named_steps["model"]
    preprocessing = pipeline[:-1]

    X_transformed = preprocessing.transform(X_sample)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    try:
        feature_names = preprocessing.get_feature_names_out()
    except (AttributeError, ValueError):
        feature_names = np.asarray(X_sample.columns)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)
    # Binary classifiers may return a list (one array per class) -- take class 1.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap.summary_plot(
        shap_values, X_transformed, feature_names=feature_names,
        show=False, max_display=20, plot_size=(10, 8),
    )
    plt.title(title, fontsize=11)
    save_fig(f"{prefix}_shap_beeswarm.png")

    shap.summary_plot(
        shap_values, X_transformed, feature_names=feature_names,
        plot_type="bar", show=False, max_display=20, plot_size=(10, 8),
    )
    plt.title(f"{title} -- mean |SHAP|", fontsize=11)
    save_fig(f"{prefix}_shap_importance.png")

    return pd.Series(
        np.abs(shap_values).mean(axis=0), index=feature_names
    ).sort_values(ascending=False)


def cost_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    amounts: np.ndarray,
    review_cost: float,
) -> pd.DataFrame:
    """Total expected cost across candidate thresholds.

    A missed fraud (false negative) costs the transaction amount -- the money
    that actually walks out the door. A false alarm costs `review_cost`, the
    price of a human looking at a legitimate transaction.
    """
    thresholds = np.unique(np.round(np.linspace(0.001, 0.999, 400), 4))
    rows = []
    for threshold in thresholds:
        predicted = y_proba >= threshold
        false_negative = (y_true == 1) & ~predicted
        false_positive = (y_true == 0) & predicted

        fraud_loss = float(amounts[false_negative].sum())
        review_spend = float(false_positive.sum() * review_cost)
        rows.append({
            "threshold": threshold,
            "fraud_loss": fraud_loss,
            "review_cost": review_spend,
            "total_cost": fraud_loss + review_spend,
            "frauds_missed": int(false_negative.sum()),
            "false_alarms": int(false_positive.sum()),
        })
    return pd.DataFrame(rows)


def plot_cost_curve(costs: pd.DataFrame, optimal: pd.Series, f1_threshold: float) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(costs["threshold"], costs["total_cost"], label="Total cost", color="#C44E52")
    ax.plot(costs["threshold"], costs["fraud_loss"], label="Fraud losses (missed)",
            alpha=0.7, linestyle="--")
    ax.plot(costs["threshold"], costs["review_cost"], label="Manual review cost",
            alpha=0.7, linestyle="--")
    ax.axvline(optimal["threshold"], color="green",
               label=f"Cost-optimal ({optimal['threshold']:.3f})")
    ax.axvline(f1_threshold, color="gray", linestyle=":",
               label=f"F1-optimal ({f1_threshold:.3f})")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Cost (EUR)")
    ax.set_title("Cost-sensitive threshold selection -- card fraud")
    ax.legend(fontsize=8)
    save_fig("20_fraud_cost_curve.png")


def explain_card(review_cost: float) -> None:
    print("\n" + "=" * 70)
    print("CARD FRAUD -- SHAP + cost-sensitive thresholds")
    print("=" * 70)

    model_path = MODEL_DIR / "card_fraud_model.joblib"
    if not model_path.exists() or not CARD_DATA.exists():
        print("Skipping -- run card_fraud.py first (model or dataset missing).")
        return

    from card_fraud import engineer_features, load_data

    bundle = joblib.load(model_path)
    pipeline, f1_threshold = bundle["pipeline"], bundle["threshold"]

    df = engineer_features(load_data(CARD_DATA))
    X, y = df.drop(columns=["Class"]), df["Class"]
    # Same split parameters as card_fraud.py, so this is the identical test set.
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    amounts = np.expm1(X_test["log_amount"].to_numpy())  # undo the log1p

    costs = cost_curve(y_test.to_numpy(), y_proba, amounts, review_cost)
    optimal = costs.loc[costs["total_cost"].idxmin()]

    do_nothing_cost = float(amounts[y_test.to_numpy() == 1].sum())
    f1_row = costs.iloc[(costs["threshold"] - f1_threshold).abs().argmin()]

    print(f"\nTotal fraud exposure in test set: EUR {do_nothing_cost:,.2f}")
    print(f"Manual review cost assumption:    EUR {review_cost:.2f} per flagged transaction")
    print(f"\nCost-optimal threshold: {optimal['threshold']:.4f}")
    print(f"  Missed frauds: {int(optimal['frauds_missed'])}  "
          f"False alarms: {int(optimal['false_alarms'])}")
    print(f"  Fraud losses:  EUR {optimal['fraud_loss']:,.2f}")
    print(f"  Review costs:  EUR {optimal['review_cost']:,.2f}")
    print(f"  TOTAL:         EUR {optimal['total_cost']:,.2f}")
    print(f"\nF1-optimal threshold ({f1_threshold:.4f}) for comparison:")
    print(f"  TOTAL:         EUR {f1_row['total_cost']:,.2f}")

    delta = f1_row["total_cost"] - optimal["total_cost"]
    if delta > 0:
        print(f"  -> Cost-optimising saves EUR {delta:,.2f} vs. the F1 cutoff.")
    else:
        print("  -> F1 cutoff happens to be near cost-optimal on this data.")
    print(f"\nDoing nothing (no model) would cost EUR {do_nothing_cost:,.2f}; "
          f"the model reduces this to EUR {optimal['total_cost']:,.2f} "
          f"({(1 - optimal['total_cost'] / do_nothing_cost) * 100:.1f}% reduction).")

    plot_cost_curve(costs, optimal, f1_threshold)
    costs.to_csv(REPORT_DIR / "card_fraud_cost_curve.csv", index=False)

    print("\nComputing SHAP values ...")
    sample = X_test.sample(min(SHAP_SAMPLE, len(X_test)), random_state=RANDOM_STATE)
    try:
        importance = shap_summary(
            pipeline, sample, "19_card", "Card fraud -- SHAP feature attribution"
        )
        print("\nTop 10 features by mean |SHAP|:")
        print(importance.head(10).round(4).to_string())
        importance.to_csv(REPORT_DIR / "card_shap_importance.csv")
    except Exception as exc:  # linear models have no TreeExplainer support
        print(f"SHAP skipped: {exc}")


def explain_hotel() -> None:
    print("\n" + "=" * 70)
    print("HOTEL CANCELLATION -- SHAP")
    print("=" * 70)

    model_path = MODEL_DIR / "best_model.joblib"
    if not model_path.exists():
        print("Skipping -- run train.py first (model missing).")
        return

    from data_prep import TARGET, get_feature_lists, load_and_prepare

    pipeline = joblib.load(model_path)
    df = load_and_prepare(HOTEL_DATA)
    numeric_cols, categorical_cols = get_feature_lists(df)
    X, y = df[numeric_cols + categorical_cols], df[TARGET]
    _, X_test, _, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    print("Computing SHAP values ...")
    sample = X_test.sample(min(SHAP_SAMPLE, len(X_test)), random_state=RANDOM_STATE)
    try:
        importance = shap_summary(
            pipeline, sample, "18_hotel", "Hotel cancellation -- SHAP feature attribution"
        )
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
