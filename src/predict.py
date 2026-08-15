"""Score new/unseen bookings with the trained cancellation model.

Input CSV should have the same raw columns as hotel_bookings.csv, minus
is_canceled/reservation_status/reservation_status_date (those won't exist
yet for a booking that hasn't happened).

Usage:
    python src/predict.py --input-path data/raw/new_bookings.csv
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from data_prep import get_feature_lists, load_and_prepare

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "best_model.joblib"
DEFAULT_OUTPUT_PATH = ROOT / "reports" / "predictions.csv"

# Columns carried through to the output purely for readability/context --
# not used as model inputs beyond what's already in the feature lists.
ID_COLS = [
    "hotel", "arrival_date_year", "arrival_date_month", "arrival_date_day_of_month",
    "lead_time", "adr", "country",
]


def score(input_path: Path, model_path: Path) -> pd.DataFrame:
    raw_row_count = len(pd.read_csv(input_path))
    df = load_and_prepare(input_path, require_target=False)
    dropped = raw_row_count - len(df)
    if dropped > 0:
        print(f"Skipped {dropped} row(s) with invalid guest/night counts (0 guests or 0 nights).")

    model = joblib.load(model_path)
    numeric_cols, categorical_cols = get_feature_lists(df)
    X = df[numeric_cols + categorical_cols]

    probabilities = model.predict_proba(X)[:, 1]
    predictions = model.predict(X)

    output = df[[c for c in ID_COLS if c in df.columns]].copy()
    output["cancellation_probability"] = probabilities.round(4)
    output["predicted_cancellation"] = predictions

    return output


def main(input_path: Path, output_path: Path, model_path: Path) -> None:
    output = score(input_path, model_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"Scored {len(output)} bookings. Predicted cancellation rate: {output['predicted_cancellation'].mean():.3f}")
    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()
    main(args.input_path, args.output_path, args.model_path)
