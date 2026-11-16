"""Fraud / abuse triage for hotel bookings.

IMPORTANT: hotel_bookings.csv has no ground-truth fraud label, so nothing
here is a trained fraud classifier and none of these scores are validated
against known fraud. This is an unsupervised triage tool: it combines
hand-written heuristics for known booking-abuse patterns with an Isolation
Forest that flags statistically unusual bookings. Output is a ranked review
queue for a human, not a fraud verdict.

Usage:
    python src/fraud_detection.py
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline

from data_prep import get_feature_lists, load_and_prepare
from modeling import RANDOM_STATE, build_preprocessor
from plotting import add_show_argument, emit, plt, report_saved

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = ROOT / "data" / "raw" / "hotel_bookings.csv"
OUTPUT_PATH = ROOT / "reports" / "fraud_review_queue.csv"

RULE_LABELS = {
    "flag_last_minute_high_value": "Skupa rezervacija u posljednji čas, bez depozita",
    "flag_serial_canceller": "Serijski otkazivač",
    "flag_excessive_changes": "Prekomjerne izmjene rezervacije (≥ 5)",
    "flag_group_rate_mismatch": "Velika grupa uz neobično nisku cijenu",
    "flag_suspicious_zero_rate": "Sumnjiva nulta cijena",
    "flag_long_unsecured_stay": "Dug boravak bez depozita i agenta",
}

# Share of bookings Isolation Forest treats as anomalous. Not an estimate of
# the real fraud rate (unknown) -- it's a review-capacity dial.
CONTAMINATION = 0.02


def add_rule_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Flag known booking-abuse patterns. Each rule is a weak signal on its own."""
    df = df.copy()

    # Chargeback/stolen-card pattern: expensive, last-minute, no deposit taken.
    df["flag_last_minute_high_value"] = (
        (df["lead_time"] <= 2)
        & (df["adr"] > df["adr"].quantile(0.95))
        & (df["deposit_type"] == "No Deposit")
    ).astype(int)

    # Serial canceller: many prior cancellations, few or no honored stays.
    df["flag_serial_canceller"] = (
        (df["previous_cancellations"] >= 3)
        & (df["previous_bookings_not_canceled"] == 0)
    ).astype(int)

    # Booking churn: repeatedly amended reservations are used to hold
    # inventory or game rate changes.
    df["flag_excessive_changes"] = (df["booking_changes"] >= 5).astype(int)

    # Unusually large party at an unusually low nightly rate.
    df["flag_group_rate_mismatch"] = (
        (df["total_guests"] >= 4) & (df["adr"] < df["adr"].quantile(0.10))
    ).astype(int)

    # Zero-rate bookings that aren't the legitimate complimentary/group segments.
    df["flag_suspicious_zero_rate"] = (
        (df["adr"] == 0)
        & (~df["market_segment"].isin(["Complementary", "Groups", "Corporate"]))
    ).astype(int)

    # Long stay booked far ahead with no deposit and no agent -- high exposure
    # if it turns out to be fraudulent.
    df["flag_long_unsecured_stay"] = (
        (df["total_nights"] >= 14)
        & (df["deposit_type"] == "No Deposit")
        & (df["has_agent"] == 0)
    ).astype(int)

    flag_cols = [c for c in df.columns if c.startswith("flag_")]
    df["rule_score"] = df[flag_cols].sum(axis=1)

    return df


def add_anomaly_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Isolation Forest over the standard feature set to catch odd combinations."""
    df = df.copy()
    numeric_cols, categorical_cols = get_feature_lists(df)
    # Exclude engineered rule columns so the model doesn't just relearn the rules.
    numeric_cols = [c for c in numeric_cols if not c.startswith(("flag_", "rule_"))]

    model = Pipeline([
        ("preprocessor", build_preprocessor(numeric_cols, categorical_cols)),
        ("iforest", IsolationForest(
            n_estimators=200,
            contamination=CONTAMINATION,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

    X = df[numeric_cols + categorical_cols]
    model.fit(X)

    # Transform once and reuse -- the encoded matrix is expensive to rebuild.
    X_encoded = model.named_steps["preprocessor"].transform(X)
    iforest = model.named_steps["iforest"]

    # decision_function: lower = more anomalous. Invert so higher = more suspicious.
    df["anomaly_score"] = -iforest.decision_function(X_encoded)
    df["is_anomaly"] = (iforest.predict(X_encoded) == -1).astype(int)

    return df


def combine_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Blend rule hits and anomaly score into one 0-100 triage priority."""
    df = df.copy()

    # Min-max both components so neither dominates purely by scale.
    anomaly_norm = (df["anomaly_score"] - df["anomaly_score"].min()) / (
        df["anomaly_score"].max() - df["anomaly_score"].min()
    )
    max_rules = df["rule_score"].max()
    rule_norm = df["rule_score"] / max_rules if max_rules > 0 else 0

    # Rules weighted higher: they encode domain meaning, while the anomaly
    # score only means "statistically unusual", which is not the same as fraud.
    df["risk_score"] = (100 * (0.6 * rule_norm + 0.4 * anomaly_norm)).round(2)

    return df


def plot_risk_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["risk_score"], bins=50, color="#C44E52")
    ax.set_xlabel("Ocjena rizika (0–100)")
    ax.set_ylabel("Broj rezervacija (logaritamska skala)")
    ax.set_yscale("log")
    ax.set_title("Raspodjela ocjene rizika sumnjivih rezervacija")
    emit(fig, "10_fraud_risk_distribution.png")


def plot_rule_hit_counts(df: pd.DataFrame) -> None:
    flag_cols = [c for c in df.columns if c.startswith("flag_")]
    counts = df[flag_cols].sum().sort_values().rename(index=RULE_LABELS)
    fig, ax = plt.subplots(figsize=(8, 4))
    counts.plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_xlabel("Broj označenih rezervacija")
    ax.set_title("Koliko često se aktivira svako pravilo")
    emit(fig, "11_fraud_rule_hits.png")


def main(data_path: Path, top_n: int) -> None:
    print(f"Loading data from {data_path} ...")
    df = load_and_prepare(data_path, require_target=False)

    df = add_rule_flags(df)
    df = add_anomaly_scores(df)
    df = combine_scores(df)

    flag_cols = [c for c in df.columns if c.startswith("flag_")]
    print(f"\nRule hit counts across {len(df)} bookings:")
    for col in flag_cols:
        print(f"  {col:<32} {int(df[col].sum()):>6}  ({df[col].mean() * 100:.2f}%)")
    print(f"\nBookings hitting >=1 rule: {(df['rule_score'] > 0).sum()} "
          f"({(df['rule_score'] > 0).mean() * 100:.2f}%)")
    print(f"Bookings flagged anomalous: {df['is_anomaly'].sum()} "
          f"({df['is_anomaly'].mean() * 100:.2f}%)")

    review_cols = [
        "hotel", "arrival_date_year", "arrival_date_month", "arrival_date_day_of_month",
        "country", "lead_time", "adr", "total_nights", "total_guests",
        "deposit_type", "market_segment", "previous_cancellations", "booking_changes",
        *flag_cols, "rule_score", "anomaly_score", "is_anomaly", "risk_score",
    ]
    queue = df[[c for c in review_cols if c in df.columns]].sort_values(
        "risk_score", ascending=False
    ).head(top_n)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    queue.to_csv(OUTPUT_PATH, index=False)

    plot_risk_distribution(df)
    plot_rule_hit_counts(df)

    print(f"\nTop {top_n} highest-risk bookings written to {OUTPUT_PATH}")
    print(report_saved())
    print("\nNOTE: no ground-truth fraud labels exist for this dataset, so these "
          "scores are unvalidated triage signals for human review, not fraud verdicts.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--top-n", type=int, default=500)
    add_show_argument(parser)
    args = parser.parse_args()
    main(args.data_path, args.top_n)
