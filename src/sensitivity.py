"""How much do duplicates and post-booking features inflate the results?

Retrains XGBoost (with the tuned hyperparameters, if tuning.py has been run)
on variants of the cancellation data and reports each variant's test score.
Variant 1 is the main setup used everywhere else; every other variant
changes one thing relative to it:

  1. main setup: duplicates removed, post-booking features excluded,
     stratified random 80/20 split
  2. post-booking features added (room_type_changed, required_car_parking_spaces)
  3. duplicates kept
  4. duplicates kept AND post-booking features added -- the project's earlier
     setup, which reported ROC-AUC 0.952
  5. country removed
  6. booking_changes and days_in_waiting_list removed: their values keep
     accumulating until arrival or cancellation, so the final value is not
     known at booking time either
  7. temporal split: train on arrivals in 2015-2016, test on 2017 -- closest
     to real use, where a model always predicts the future

ROC-AUC is the metric compared, because it does not depend on the share of
cancellations, which differs between variants (37% with duplicates, 28%
without). PR-AUC is reported too, but is not comparable across variants.

Usage:
    python src/sensitivity.py
"""

import argparse
import warnings
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score

from data_prep import load_modeling_data
from modeling import build_pipeline, load_params, positive_scores, split_data
from plotting import add_show_argument, emit, num, plt, report_saved

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = ROOT / "data" / "raw" / "hotel_bookings.csv"
REPORT_DIR = ROOT / "reports"
PARAMS_PATH = REPORT_DIR / "hotel_best_params.json"

warnings.filterwarnings("ignore")

VARIANTS = [
    ("main", "1. Glavni model (bez duplikata, bez obilježja s dolaska)", {}),
    ("post_booking", "2. + obilježja zabilježena pri dolasku gosta",
     {"include_post_booking": True}),
    ("duplicates", "3. Duplikati zadržani", {"deduplicate": False}),
    ("earlier_setup", "4. Duplikati + obilježja s dolaska (ranija verzija)",
     {"deduplicate": False, "include_post_booking": True}),
    ("no_country", "5. Bez obilježja country", {"drop": ("country",)}),
    ("strict", "6. Bez booking_changes i days_in_waiting_list",
     {"drop": ("booking_changes", "days_in_waiting_list")}),
    ("temporal", "7. Vremenska podjela (trening 2015–2016, test 2017)", {"temporal": True}),
]


def run_variant(data_path: Path, params: dict | None, deduplicate: bool = True,
                include_post_booking: bool = False, drop: tuple = (),
                temporal: bool = False) -> dict:
    X, y, numeric_cols, categorical_cols = load_modeling_data(
        data_path, deduplicate=deduplicate, include_post_booking=include_post_booking)
    numeric_cols = [c for c in numeric_cols if c not in drop]
    categorical_cols = [c for c in categorical_cols if c not in drop]

    if temporal:
        is_train = X["arrival_date_year"] < 2017
        # The test year never occurs in training, so the year itself is useless.
        categorical_cols = [c for c in categorical_cols if c != "arrival_date_year"]
        X_train, X_test, y_train, y_test = X[is_train], X[~is_train], y[is_train], y[~is_train]
    else:
        X_train, X_test, y_train, y_test = split_data(X, y)

    columns = numeric_cols + categorical_cols
    pipeline = build_pipeline("xgboost", "hotel", numeric_cols, categorical_cols, y_train, params)
    pipeline.fit(X_train[columns], y_train)
    scores = positive_scores(pipeline, X_test[columns])
    predictions = pipeline.predict(X_test[columns])
    return {
        "test_roc_auc": roc_auc_score(y_test, scores),
        "test_pr_auc": average_precision_score(y_test, scores),
        "test_f1": f1_score(y_test, predictions),
        "test_accuracy": accuracy_score(y_test, predictions),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "test_cancellation_rate": y_test.mean(),
    }


def plot_sensitivity(results: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ordered = results.iloc[::-1]
    colors = ["#4C72B0" if key == "main" else "#9FB4D3" for key in ordered.index]
    ax.barh(ordered["variant"], ordered["test_roc_auc"], color=colors)
    for y_pos, value in enumerate(ordered["test_roc_auc"]):
        ax.annotate(f" {num(value)}", (value, y_pos), va="center", fontsize=9)
    main_auc = results.loc["main", "test_roc_auc"]
    ax.axvline(main_auc, color="black", linestyle=":", linewidth=1)
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("ROC-AUC na testnom skupu (XGBoost)")
    ax.set_title("Osjetljivost rezultata na duplikate, obilježja s dolaska i način podjele")
    emit(fig, "30_leakage_sensitivity.png")


def main(data_path: Path) -> None:
    params = load_params(PARAMS_PATH).get("xgboost")
    print("Using tuned XGBoost hyperparameters." if params else
          "No tuned hyperparameters found -- using defaults.")

    rows = {}
    for key, label, options in VARIANTS:
        print(f"\n--- {label}")
        rows[key] = {"variant": label, **run_variant(data_path, params, **options)}
        print(f"    test ROC-AUC {rows[key]['test_roc_auc']:.4f}")

    results = pd.DataFrame(rows).T
    print("\n" + results[["test_roc_auc", "test_pr_auc", "test_f1", "n_train", "n_test"]]
          .astype(float).round(4).to_string())

    plot_sensitivity(results)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(REPORT_DIR / "hotel_leakage_sensitivity.csv")
    print(f"\nSaved {REPORT_DIR / 'hotel_leakage_sensitivity.csv'}")
    print(report_saved())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    add_show_argument(parser)
    args = parser.parse_args()
    main(args.data_path)
