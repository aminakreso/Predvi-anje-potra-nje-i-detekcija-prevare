"""Load and prepare the Hotel Booking Demand dataset for cancellation prediction.

Dataset: Antonio, Almeida & Nunes (2019), "Hotel booking demand datasets",
available on Kaggle as hotel_bookings.csv. Target column: is_canceled.

Every cleaning rule here is a fixed, row-by-row rule (no statistics computed
over the whole dataset), so preparing the data before the train/test split
cannot leak information from the test rows. Rules that need statistics --
such as merging rare countries -- live inside the model pipeline instead
(see modeling.build_preprocessor), where they are learned on training data.
"""

from pathlib import Path

import pandas as pd

TARGET = "is_canceled"

# These columns are only known once a stay is over / a cancellation has
# happened, so they leak the target if used as predictors.
LEAKAGE_COLS = ["reservation_status", "reservation_status_date"]

# Recorded or updated only when the guest arrives, so their final values are
# not known when the booking is made:
#   - room_type_changed compares the reserved room with the ASSIGNED room,
#     and the room is assigned at check-in;
#   - required_car_parking_spaces: not one booking with a parking space in
#     this dataset was canceled, which only makes sense if the field is
#     filled in on arrival.
# They are excluded from the models by default; sensitivity.py measures how
# much they would inflate the results.
POST_BOOKING_FEATURES = ["room_type_changed", "required_car_parking_spaces"]

# Highest plausible average daily rate. The dataset has a single booking at
# 5,400 (the next highest is 510), which is a data-entry error.
MAX_VALID_ADR = 1000


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

    # A negative or absurdly high nightly rate is a data-entry error.
    df = df[df["adr"].between(0, MAX_VALID_ADR)]

    df = df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns])

    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]
    df["is_family"] = ((df["children"] + df["babies"]) > 0).astype(int)

    # Whether the guest ended up in a different room type than reserved.
    # Kept for analysis, but excluded from the models: see POST_BOOKING_FEATURES.
    df["room_type_changed"] = (
        df["reserved_room_type"] != df["assigned_room_type"]
    ).astype(int)

    df["has_agent"] = (df["agent"] != "0").astype(int)
    df["has_company"] = (df["company"] != "0").astype(int)

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


def get_feature_lists(
    df: pd.DataFrame, include_post_booking: bool = False
) -> tuple[list[str], list[str]]:
    """Return (numeric_cols, categorical_cols) for the model features."""
    drop_cols = {TARGET, "agent", "company", "reserved_room_type", "assigned_room_type"}
    if not include_post_booking:
        drop_cols |= set(POST_BOOKING_FEATURES)
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


def load_modeling_data(
    path: str | Path,
    deduplicate: bool = True,
    include_post_booking: bool = False,
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """Prepared features and target for the cancellation classifiers.

    About 28% of the cleaned rows are exact copies of another row -- identical
    in every model feature and in the target (typically several rooms of one
    group booking). With a random split the copies land on both sides, and the
    model is scored on rows it has effectively already seen. Duplicates are
    therefore removed before splitting. The demand series and the fraud
    triage keep them, because there every row is a real booking to count.
    """
    df = load_and_prepare(path)
    numeric_cols, categorical_cols = get_feature_lists(df, include_post_booking)
    feature_cols = numeric_cols + categorical_cols

    if deduplicate:
        n_before = len(df)
        df = df.drop_duplicates(subset=feature_cols + [TARGET]).reset_index(drop=True)
        print(f"Removed {n_before - len(df)} duplicate rows; {len(df)} remain "
              f"(cancellation rate {df[TARGET].mean():.3f}).")

    return df[feature_cols], df[TARGET], numeric_cols, categorical_cols


if __name__ == "__main__":
    default_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "hotel_bookings.csv"
    data = load_and_prepare(default_path)
    numeric_cols, categorical_cols = get_feature_lists(data)
    print(f"Rows after cleaning: {len(data)}")
    print(f"Cancellation rate: {data[TARGET].mean():.3f}")
    print(f"Numeric features ({len(numeric_cols)}): {numeric_cols}")
    print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")
    load_modeling_data(default_path)  # prints how many duplicates are removed
