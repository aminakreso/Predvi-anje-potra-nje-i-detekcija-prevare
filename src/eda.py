"""Exploratory data analysis for the hotel booking cancellation dataset.

Saves a handful of figures to reports/figures/ instead of relying on
notebook cell output, so results survive outside a notebook session.
"""

import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns

from data_prep import TARGET, load_and_prepare
from plotting import FIG_DIR, add_show_argument, emit, plt, report_saved

sns.set_theme(style="whitegrid")

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "hotel_bookings.csv"


def save(fig, name: str) -> None:
    emit(fig, name)


def plot_cancellation_rate_by(df: pd.DataFrame, column: str, filename: str, rotate: int = 0) -> None:
    rate = df.groupby(column)[TARGET].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    rate.plot(kind="bar", ax=ax, color="#4C72B0")
    ax.set_ylabel("Cancellation rate")
    ax.set_title(f"Cancellation rate by {column}")
    plt.xticks(rotation=rotate, ha="right" if rotate else "center")
    save(fig, filename)


def plot_overall_rate(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    df[TARGET].value_counts(normalize=True).sort_index().plot(
        kind="bar", ax=ax, color=["#55A868", "#C44E52"]
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Not canceled", "Canceled"], rotation=0)
    ax.set_ylabel("Share of bookings")
    ax.set_title("Overall cancellation rate")
    save(fig, "01_overall_cancellation_rate.png")


def plot_lead_time(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df, x=TARGET, y="lead_time", ax=ax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Not canceled", "Canceled"])
    ax.set_title("Lead time vs. cancellation")
    save(fig, "05_lead_time_vs_cancellation.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()[[TARGET]].sort_values(by=TARGET, ascending=False)
    fig, ax = plt.subplots(figsize=(4, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title(f"Correlation with {TARGET}")
    save(fig, "06_correlation_with_target.png")


def plot_monthly_trend(df: pd.DataFrame) -> None:
    monthly = df.groupby("arrival_month_num")[TARGET].mean().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    monthly.plot(kind="line", marker="o", ax=ax, color="#4C72B0")
    ax.set_xlabel("Arrival month")
    ax.set_ylabel("Cancellation rate")
    ax.set_title("Cancellation rate by arrival month")
    save(fig, "04_monthly_cancellation_trend.png")


def run(data_path: Path = DATA_PATH) -> None:
    df = load_and_prepare(data_path)
    print(f"Loaded {len(df)} rows after cleaning.")

    plot_overall_rate(df)
    plot_cancellation_rate_by(df, "hotel", "02_cancellation_by_hotel.png")
    plot_cancellation_rate_by(df, "deposit_type", "03_cancellation_by_deposit_type.png")
    plot_monthly_trend(df)
    plot_lead_time(df)
    plot_correlation_heatmap(df)

    print(report_saved())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_show_argument(parser)
    parser.parse_args()
    run()
