"""Exploratory data analysis for both datasets.

Saves figures to reports/figures/ instead of relying on notebook cell output,
so results survive outside a notebook session. All labels are in Bosnian, to
go straight into the thesis.

Hotel figures describe all cleaned bookings (duplicates included, since each
row is a real booking); the classifiers additionally remove duplicates -- see
data_prep.load_modeling_data.

Usage:
    python src/eda.py
    python src/eda.py --task card
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns

from data_prep import TARGET, load_and_prepare, load_data
from plotting import add_show_argument, emit, num, plt, report_saved

sns.set_theme(style="whitegrid")

ROOT = Path(__file__).resolve().parents[1]
HOTEL_DATA = ROOT / "data" / "raw" / "hotel_bookings.csv"
CARD_DATA = ROOT / "data" / "raw" / "creditcard.csv"

STATUS_LABELS = ["Nije otkazana", "Otkazana"]
MONTHS_BS = ["Jan", "Feb", "Mar", "Apr", "Maj", "Jun", "Jul", "Avg", "Sep", "Okt", "Nov", "Dec"]
CANCEL_AXIS = "Udio otkazanih rezervacija (%)"
COLUMN_LABELS = {
    "hotel": "Hotel",
    "deposit_type": "Vrsta depozita",
    "market_segment": "Tržišni segment",
    "distribution_channel": "Kanal distribucije",
    "customer_type": "Tip gosta",
}


def annotate_counts(ax, counts, show_value: bool = False) -> None:
    """Write the number of bookings above each bar (small groups mislead)."""
    for patch, count in zip(ax.patches, counts):
        text = f"n={count:,}".replace(",", ".")
        if show_value:
            text = f"{num(patch.get_height(), 1)} %\n{text}"
        ax.annotate(text, (patch.get_x() + patch.get_width() / 2, patch.get_height()),
                    ha="center", va="bottom", fontsize=8)


MIN_GROUP_SIZE = 100  # a rate from a handful of bookings (e.g. "Undefined", n=2) is noise


def plot_cancellation_rate_by(df: pd.DataFrame, column: str, filename: str, rotate: int = 0) -> None:
    grouped = df.groupby(column)[TARGET].agg(["mean", "size"]).sort_values("mean", ascending=False)
    grouped = grouped[grouped["size"] >= MIN_GROUP_SIZE]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    (grouped["mean"] * 100).plot(kind="bar", ax=ax, color="#4C72B0")
    annotate_counts(ax, grouped["size"])
    ax.set_xlabel(COLUMN_LABELS.get(column, column))
    ax.set_ylabel(CANCEL_AXIS)
    ax.set_title(f"Udio otkazivanja po obilježju {column}")
    plt.setp(ax.get_xticklabels(), rotation=rotate, ha="right" if rotate else "center",
             rotation_mode="anchor")
    emit(fig, filename)


def plot_overall_rate(df: pd.DataFrame) -> None:
    shares = df[TARGET].value_counts(normalize=True).sort_index() * 100
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    shares.plot(kind="bar", ax=ax, color=["#55A868", "#C44E52"])
    for patch, share in zip(ax.patches, shares):
        ax.annotate(f"{num(share, 1)} %", (patch.get_x() + patch.get_width() / 2, patch.get_height()),
                    ha="center", va="bottom")
    ax.set_xticks([0, 1], STATUS_LABELS, rotation=0)
    ax.set_xlabel("Status rezervacije")
    ax.set_ylabel("Udio rezervacija (%)")
    ax.set_title("Raspodjela ciljne varijable is_canceled")
    emit(fig, "01_overall_cancellation_rate.png")


def plot_monthly_trend(df: pd.DataFrame) -> None:
    monthly = df.groupby("arrival_month_num")[TARGET].mean().sort_index() * 100
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(monthly.index, monthly.values, marker="o", color="#4C72B0")
    ax.set_xticks(range(1, 13), MONTHS_BS)
    ax.set_xlabel("Mjesec dolaska")
    ax.set_ylabel(CANCEL_AXIS)
    ax.set_title("Udio otkazivanja po mjesecu dolaska")
    emit(fig, "04_monthly_cancellation_trend.png")


def plot_lead_time(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.boxplot(data=df, x=TARGET, y="lead_time", ax=ax)
    ax.set_xticks([0, 1], STATUS_LABELS)
    ax.set_xlabel("Status rezervacije")
    ax.set_ylabel("Vrijeme najave (dani od rezervacije do dolaska)")
    ax.set_title("Vrijeme najave i otkazivanje")
    emit(fig, "05_lead_time_vs_cancellation.png")


def plot_correlation_with_target(df: pd.DataFrame) -> None:
    corr = df.select_dtypes(include="number").corr()[[TARGET]].drop(index=TARGET)
    corr = corr.sort_values(by=TARGET, ascending=False)
    labels = corr.map(lambda v: num(v, 2))
    fig, ax = plt.subplots(figsize=(4.5, 8))
    sns.heatmap(corr, annot=labels, fmt="", cmap="coolwarm", center=0, ax=ax,
                cbar_kws={"label": "Pearsonov koeficijent korelacije"})
    ax.set_xticklabels([TARGET])
    ax.set_title("Korelacija numeričkih obilježja s is_canceled")
    emit(fig, "06_correlation_with_target.png")


def plot_correlation_matrix(df: pd.DataFrame) -> None:
    corr = df.select_dtypes(include="number").corr()
    labels = corr.map(lambda v: num(v, 2))
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, annot=labels, fmt="", cmap="coolwarm", center=0, vmin=-1, vmax=1,
                ax=ax, annot_kws={"size": 6}, cbar_kws={"label": "Pearsonov koeficijent korelacije"})
    ax.set_title("Korelaciona matrica numeričkih obilježja")
    emit(fig, "25_correlation_matrix.png")


def plot_missing_values(raw: pd.DataFrame) -> None:
    missing = raw.isna().mean().mul(100)
    missing = missing[missing > 0].sort_values()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    missing.plot(kind="barh", ax=ax, color="#8172B2")
    for patch, value, count in zip(ax.patches, missing, raw[missing.index].isna().sum()):
        ax.annotate(f" {num(value, 2)} % ({count:,})".replace(",", "."),
                    (patch.get_width(), patch.get_y() + patch.get_height() / 2), va="center", fontsize=9)
    ax.set_xlim(0, 110)
    ax.set_xlabel("Udio nedostajućih vrijednosti (%)")
    ax.set_ylabel("Kolona")
    ax.set_title(f"Nedostajuće vrijednosti u izvornom skupu ({len(raw):,} redova, "
                 f"{raw.shape[1]} kolona)".replace(",", "."))
    emit(fig, "21_missing_values.png")


def plot_monthly_bookings(df: pd.DataFrame) -> None:
    period = pd.to_datetime(dict(year=df["arrival_date_year"], month=df["arrival_month_num"], day=1))
    counts = df.groupby([period, TARGET]).size().unstack(fill_value=0).sort_index()
    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = np.arange(len(counts))
    ax.bar(x, counts[0], color="#55A868", label="Realizovane")
    ax.bar(x, counts[1], bottom=counts[0], color="#C44E52", label="Otkazane")
    ax.set_xticks(x, [f"{MONTHS_BS[d.month - 1]} {d.year}" for d in counts.index])
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right", rotation_mode="anchor")
    ax.set_xlabel("Mjesec dolaska")
    ax.set_ylabel("Broj rezervacija")
    ax.set_title("Broj rezervacija po mjesecu dolaska – ukupna i realizovana potražnja")
    ax.legend()
    emit(fig, "31_monthly_bookings.png")


def plot_requests_and_history(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    requests = df.groupby("total_of_special_requests")[TARGET].agg(["mean", "size"])
    (requests["mean"] * 100).plot(kind="bar", ax=axes[0], color="#4C72B0")
    annotate_counts(axes[0], requests["size"])
    axes[0].set_xlabel("Broj posebnih zahtjeva")
    axes[0].set_ylabel(CANCEL_AXIS)
    axes[0].set_title("Posebni zahtjevi gosta")
    axes[0].tick_params(axis="x", rotation=0)

    history = pd.cut(df["previous_cancellations"], [-1, 0, 1, np.inf], labels=["0", "1", "2 i više"])
    previous = df.groupby(history, observed=True)[TARGET].agg(["mean", "size"])
    (previous["mean"] * 100).plot(kind="bar", ax=axes[1], color="#DD8452")
    annotate_counts(axes[1], previous["size"])
    axes[1].set_xlabel("Broj ranijih otkazivanja gosta")
    axes[1].set_ylabel(CANCEL_AXIS)
    axes[1].set_title("Historija otkazivanja")
    axes[1].tick_params(axis="x", rotation=0)

    fig.suptitle("Udio otkazivanja prema ponašanju gosta")
    emit(fig, "32_cancellation_by_requests_and_history.png")


def plot_post_booking_features(df: pd.DataFrame) -> None:
    """The evidence for excluding these two columns from the models."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    panels = [
        (df["room_type_changed"].map({0: "Ista kao rezervisana", 1: "Različita"}),
         "Dodijeljena soba", ["Ista kao rezervisana", "Različita"]),
        (np.where(df["required_car_parking_spaces"] > 0, "Traženo", "Nije traženo"),
         "Parking mjesto", ["Nije traženo", "Traženo"]),
    ]
    for ax, (groups, label, order) in zip(axes, panels):
        stats = df.groupby(groups)[TARGET].agg(["mean", "size"]).reindex(order)
        (stats["mean"] * 100).plot(kind="bar", ax=ax, color=["#4C72B0", "#C44E52"])
        annotate_counts(ax, stats["size"], show_value=True)
        ax.set_ylim(0, 50)
        ax.set_xlabel(label)
        ax.set_ylabel(CANCEL_AXIS)
        ax.tick_params(axis="x", rotation=0)
    fig.suptitle("Obilježja koja se popunjavaju tek pri dolasku gosta (isključena iz modela)")
    emit(fig, "26_post_booking_features.png")


def run_hotel() -> None:
    raw = load_data(HOTEL_DATA)
    df = load_and_prepare(HOTEL_DATA)
    print(f"Hotel: {len(raw)} raw rows, {len(df)} after cleaning.")

    plot_missing_values(raw)
    plot_overall_rate(df)
    plot_cancellation_rate_by(df, "hotel", "02_cancellation_by_hotel.png")
    plot_cancellation_rate_by(df, "deposit_type", "03_cancellation_by_deposit_type.png")
    plot_cancellation_rate_by(df, "market_segment", "22_cancellation_by_market_segment.png", rotate=30)
    plot_cancellation_rate_by(df, "distribution_channel", "23_cancellation_by_distribution_channel.png")
    plot_cancellation_rate_by(df, "customer_type", "24_cancellation_by_customer_type.png")
    plot_monthly_trend(df)
    plot_lead_time(df)
    plot_correlation_with_target(df)
    plot_correlation_matrix(df)
    plot_monthly_bookings(df)
    plot_requests_and_history(df)
    plot_post_booking_features(df)


def run_card() -> None:
    if not CARD_DATA.exists():
        print(f"Skipping card EDA -- {CARD_DATA} not found.")
        return
    df = pd.read_csv(CARD_DATA)
    print(f"Card: {len(df)} transactions, {int(df['Class'].sum())} fraudulent.")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for cls, label, color in [(0, "Legitimne", "#55A868"), (1, "Prevare", "#C44E52")]:
        ax.hist(np.log1p(df.loc[df["Class"] == cls, "Amount"]), bins=50, density=True,
                alpha=0.6, label=label, color=color)
    ax.set_xlabel("log(1 + iznos transakcije u EUR)")
    ax.set_ylabel("Gustina")
    ax.set_title("Raspodjela iznosa transakcija po klasi")
    ax.legend()
    emit(fig, "27_card_amount_by_class.png")

    hour = (df["Time"] // 3600) % 24
    rate = df.groupby(hour)["Class"].mean() * 100
    fig, ax = plt.subplots(figsize=(10, 4.5))
    rate.plot(kind="bar", ax=ax, color="#C44E52")
    ax.axhline(df["Class"].mean() * 100, color="gray", linestyle="--",
               label=f"Prosjek ({num(df['Class'].mean() * 100, 3)} %)")
    ax.set_xlabel("Sat u danu")
    ax.set_ylabel("Udio prevarnih transakcija (%)")
    ax.set_title("Udio prevara po satu u danu")
    ax.tick_params(axis="x", rotation=0)
    ax.legend()
    emit(fig, "28_card_fraud_rate_by_hour.png")


def run(task: str) -> None:
    if task in ("hotel", "both"):
        run_hotel()
    if task in ("card", "both"):
        run_card()
    print(report_saved())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["hotel", "card", "both"], default="both")
    add_show_argument(parser)
    args = parser.parse_args()
    run(args.task)
