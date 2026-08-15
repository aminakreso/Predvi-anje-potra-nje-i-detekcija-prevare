"""Load and prepare the Hotel Booking Demand dataset for cancellation prediction.

Dataset: Antonio, Almeida & Nunes (2019), "Hotel booking demand datasets",
available on Kaggle as hotel_bookings.csv. Target column: is_canceled.
"""

from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "is_canceled"

# These columns are only known once a stay is over / a cancellation has
# happened, so they leak the target if used as predictors.
LEAKAGE_COLS = ["reservation_status", "reservation_status_date"]

RARE_COUNTRY_THRESHOLD = 0.01  # countries below 1% of bookings -> "Other"


def load_data(path: str | Path, require_target: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path)
    if require_target and TARGET not in df.columns:
        raise ValueError(f"Expected target column '{TARGET}' not found in {path}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Missing-value handling (documented causes in the original dataset).
    df["children"] = df["children"].fillna(0)
    df["country"] = df["country"].fillna("Unknown")
    df["agent"] = df["agent"].fillna(0).astype(int).astype(str)
    df["company"] = df["company"].fillna(0).astype(int).astype(str)

    # Drop bookings with zero total guests -- these are data-entry errors,
    # not real reservations.
    total_guests = df["adults"] + df["children"] + df["babies"]
    df = df[total_guests > 0]

    # adr (average daily rate) has a couple of extreme/negative outliers
    # (e.g. one row at 5400, some at 0 or negative) that don't represent
    # real pricing -- clip to a sane range instead of dropping rows.
    df["adr"] = df["adr"].clip(lower=0, upper=df["adr"].quantile(0.999))

    df = df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns])

    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]
    df["is_family"] = ((df["children"] + df["babies"]) > 0).astype(int)

    # Whether the guest ended up in a different room type than reserved --
    # a signal that shows up in several of the reference notebooks.
    df["room_type_changed"] = (
        df["reserved_room_type"] != df["assigned_room_type"]
    ).astype(int)

    df["has_agent"] = (df["agent"] != "0").astype(int)
    df["has_company"] = (df["company"] != "0").astype(int)

    # Collapse rare countries so one-hot encoding doesn't explode into
    # ~170 sparse columns.
    country_freq = df["country"].value_counts(normalize=True)
    rare_countries = country_freq[country_freq < RARE_COUNTRY_THRESHOLD].index
    df["country"] = df["country"].where(~df["country"].isin(rare_countries), "Other")

    # Month name -> number for a numeric ordering signal.
    month_map = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12,
    }
    df["arrival_month_num"] = df["arrival_date_month"].map(month_map)

    # Total nights of 0 is another data-entry edge case; drop it here
    # rather than in clean_data since it depends on an engineered column.
    df = df[df["total_nights"] > 0].reset_index(drop=True)

    return df


FORCE_CATEGORICAL = {
    "arrival_date_year", "is_repeated_guest", "is_family",
    "has_agent", "has_company", "room_type_changed",
}


def get_feature_lists(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (numeric_cols, categorical_cols) for the model features."""
    drop_cols = {TARGET, "agent", "company", "reserved_room_type", "assigned_room_type"}
    feature_cols = [c for c in df.columns if c not in drop_cols]

    # Use is_numeric_dtype rather than comparing to `object` directly --
    # pandas >=2.x can back string columns with a dedicated string dtype
    # (not `object`), which a plain `dtype == object` check would miss.
    categorical_cols = [
        c for c in feature_cols
        if c in FORCE_CATEGORICAL or not pd.api.types.is_numeric_dtype(df[c])
    ]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    return numeric_cols, categorical_cols


def load_and_prepare(path: str | Path, require_target: bool = True) -> pd.DataFrame:
    df = load_data(path, require_target=require_target)
    df = clean_data(df)
    df = engineer_features(df)
    return df


if __name__ == "__main__":
    default_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "hotel_bookings.csv"
    data = load_and_prepare(default_path)
    numeric_cols, categorical_cols = get_feature_lists(data)
    print(f"Rows after cleaning: {len(data)}")
    print(f"Cancellation rate: {data[TARGET].mean():.3f}")
    print(f"Numeric features ({len(numeric_cols)}): {numeric_cols}")
    print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")
