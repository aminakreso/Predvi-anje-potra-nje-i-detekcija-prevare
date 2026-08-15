"""Shared pytest fixtures.

Unit tests run against a small synthetic frame so they stay fast; the few
integration tests that need real data use the session-scoped `real_hotel_df`
fixture and skip cleanly when the CSV is absent.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

HOTEL_DATA = ROOT / "data" / "raw" / "hotel_bookings.csv"

MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]


@pytest.fixture
def raw_hotel_df() -> pd.DataFrame:
    """Synthetic frame with the same schema as hotel_bookings.csv.

    Deliberately includes the edge cases clean_data is supposed to remove:
    a zero-guest booking (row 3) and a zero-night booking (row 4).
    """
    rng = np.random.default_rng(0)
    n = 60

    df = pd.DataFrame({
        "hotel": rng.choice(["City Hotel", "Resort Hotel"], n),
        "is_canceled": rng.integers(0, 2, n),
        "lead_time": rng.integers(0, 400, n),
        "arrival_date_year": rng.choice([2015, 2016, 2017], n),
        "arrival_date_month": rng.choice(MONTHS, n),
        "arrival_date_week_number": rng.integers(1, 53, n),
        "arrival_date_day_of_month": rng.integers(1, 29, n),
        "stays_in_weekend_nights": rng.integers(0, 3, n),
        "stays_in_week_nights": rng.integers(1, 6, n),
        "adults": rng.integers(1, 3, n),
        "children": rng.choice([0.0, 1.0, np.nan], n),
        "babies": rng.integers(0, 2, n),
        "meal": rng.choice(["BB", "HB", "SC"], n),
        "country": rng.choice(["PRT", "GBR", "FRA", "ZZZ"], n),
        "market_segment": rng.choice(["Online TA", "Direct", "Groups"], n),
        "distribution_channel": rng.choice(["TA/TO", "Direct"], n),
        "is_repeated_guest": rng.integers(0, 2, n),
        "previous_cancellations": rng.integers(0, 4, n),
        "previous_bookings_not_canceled": rng.integers(0, 3, n),
        "reserved_room_type": rng.choice(["A", "B", "C"], n),
        "assigned_room_type": rng.choice(["A", "B", "C"], n),
        "booking_changes": rng.integers(0, 7, n),
        "deposit_type": rng.choice(["No Deposit", "Non Refund"], n),
        "agent": rng.choice([9.0, 240.0, np.nan], n),
        "company": rng.choice([40.0, np.nan], n),
        "days_in_waiting_list": rng.integers(0, 5, n),
        "customer_type": rng.choice(["Transient", "Contract"], n),
        "adr": rng.uniform(0, 200, n).round(2),
        "required_car_parking_spaces": rng.integers(0, 2, n),
        "total_of_special_requests": rng.integers(0, 4, n),
        "reservation_status": rng.choice(["Check-Out", "Canceled"], n),
        "reservation_status_date": "2017-01-01",
    })

    # Guarantee the edge cases exist regardless of the random draw.
    df.loc[3, ["adults", "children", "babies"]] = [0, 0.0, 0]
    df.loc[4, ["stays_in_weekend_nights", "stays_in_week_nights"]] = [0, 0]
    df.loc[4, ["adults", "children", "babies"]] = [2, 0.0, 0]

    return df


@pytest.fixture
def prepared_hotel_df(raw_hotel_df, tmp_path):
    """raw_hotel_df pushed through the real load_and_prepare pipeline."""
    from data_prep import load_and_prepare

    csv_path = tmp_path / "bookings.csv"
    raw_hotel_df.to_csv(csv_path, index=False)
    return load_and_prepare(csv_path)


@pytest.fixture(scope="session")
def real_hotel_df():
    from data_prep import load_and_prepare

    if not HOTEL_DATA.exists():
        pytest.skip(f"{HOTEL_DATA} not present")
    return load_and_prepare(HOTEL_DATA)
