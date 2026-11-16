"""Exploratory data analysis (EDA) for both datasets.

Every figure is saved to reports/figures/ with Bosnian labels, ready for the
thesis. Files are named eda01_... to eda16_..., numbered in the order the script
creates them (the same order as the list below).

HOTEL BOOKINGS (data/raw/hotel_bookings.csv)
Figures use all cleaned bookings, duplicates included, because each row is a
real booking the hotel had to handle. The models remove duplicates separately
(see data_prep.load_modeling_data).

  eda01_nedostajuce_vrijednosti                Share of missing values per column in the raw file.
  eda02_ukupni_udio_otkazivanja     Share of cancelled vs. not cancelled bookings.
  eda03_otkazivanje_po_hotelu         Cancellation rate: City Hotel vs. Resort Hotel.
  eda04_otkazivanje_po_vrsti_depozita  ... by deposit type (No Deposit / Non Refund / Refundable).
  eda05_otkazivanje_po_trzisnom_segmentu
                                      ... by market segment (Groups, Online TA, Direct, ...).
  eda06_otkazivanje_po_kanalu_distribucije
                                      ... by booking channel (TA/TO, Direct, ...).
  eda07_otkazivanje_po_tipu_gosta ... by customer type (Transient, Contract, ...).
  eda08_otkazivanje_po_mjesecu_dolaska    Cancellation rate per arrival month (Jan-Dec).
  eda09_otkazivanje_po_vremenu_najave     Cancellation rate per lead-time band (0-7 days ... > 365).
  eda10_korelacija_s_otkazivanjem       Pearson correlation of every numeric feature with is_canceled.
  eda11_korelaciona_matrica            Pearson correlations between all numeric features.
  eda12_rezervacije_po_mjesecu              Bookings per arrival month, split into realized / cancelled.
  eda13_otkazivanje_po_zahtjevima_i_historiji
                                      Cancellation rate by number of special requests (left)
                                      and by number of earlier cancellations (right).
  eda14_obiljezja_s_dolaska         Cancellation rate by room change and parking request:
                                      the evidence for leaving these two columns out of the models.

CREDIT CARD TRANSACTIONS (data/raw/creditcard.csv)
  eda15_kartice_iznos_po_klasi          Distribution of transaction amounts, legitimate vs. fraud.
  eda16_kartice_prevare_po_satu       Share of fraudulent transactions per hour of the day.

Usage:
    python src/eda.py                # both datasets
    python src/eda.py --task hotel   # only hotel bookings
    python src/eda.py --task card    # only credit cards
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns

from data_prep import TARGET, load_and_prepare, load_data
from plotting import add_show_argument, emit, num, plt, report_saved

sns.set_theme(style="whitegrid")

# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
HOTEL_DATA = ROOT / "data" / "raw" / "hotel_bookings.csv"
CARD_DATA = ROOT / "data" / "raw" / "creditcard.csv"

# --- Colors ----------------------------------------------------------------
BLUE = "#4C72B0"     # default bar color
GREEN = "#55A868"    # not cancelled / legitimate
RED = "#C44E52"      # cancelled / fraud
ORANGE = "#DD8452"
PURPLE = "#8172B2"

# --- Bosnian labels --------------------------------------------------------
STATUS_LABELS = ["Nije otkazana", "Otkazana"]
MONTHS_BS = ["Jan", "Feb", "Mar", "Apr", "Maj", "Jun", "Jul", "Avg", "Sep", "Okt", "Nov", "Dec"]
CANCEL_AXIS = "Udio otkazanih rezervacija (%)"
CORR_LABEL = "Pearsonov koeficijent korelacije"
COLUMN_LABELS = {
    "hotel": "Hotel",
    "deposit_type": "Vrsta depozita",
    "market_segment": "Tržišni segment",
    "distribution_channel": "Kanal distribucije",
    "customer_type": "Tip gosta",
}

# Groups smaller than this are left out: a rate from a handful of bookings
# (e.g. market segment "Undefined", n=2) is noise.
MIN_GROUP_SIZE = 100

# Lead-time bands for figure 05, in days between booking and arrival.
LEAD_TIME_BINS = [-1, 7, 30, 90, 180, 365, float("inf")]
LEAD_TIME_LABELS = ["0–7", "8–30", "31–90", "91–180", "181–365", "> 365"]


# =============================================================================
# Helpers
# =============================================================================

def thousands(n: int) -> str:
    """12345 -> '12.345' (Bosnian thousands separator)."""
    return f"{n:,}".replace(",", ".")


def cancellation_rate(df: pd.DataFrame, groups) -> pd.DataFrame:
    """Cancellation rate in % ('rate') and number of bookings ('n') per group."""
    stats = df.groupby(groups, observed=True)[TARGET].agg(rate="mean", n="size")
    stats["rate"] *= 100
    return stats


def label_bars(ax, counts, show_rate: bool = False) -> None:
    """Write 'n=...' above each bar (and the rate itself, if show_rate)."""
    for bar, count in zip(ax.patches, counts):
        text = f"n={thousands(count)}"
        if show_rate:
            text = f"{num(bar.get_height(), 1)} %\n{text}"
        ax.annotate(text, (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=8)


def rate_bars(ax, stats: pd.DataFrame, color=BLUE, show_rate: bool = False) -> None:
    """Bar chart of cancellation rates, with the group sizes written on top."""
    stats["rate"].plot(kind="bar", ax=ax, color=color, rot=0)
    label_bars(ax, stats["n"], show_rate)
    ax.set_ylabel(CANCEL_AXIS)


# =============================================================================
# Hotel bookings
# =============================================================================

# eda01 – Nedostajuće vrijednosti
# Zašto: prije svake analize treba znati gdje podaci nedostaju i koliko.
# Zaključak: nedostaju samo company (94 %), agent (14 %), country (0,4 %) i
# children (4 reda). Prazno polje kod company/agent znači "bez firme/agencije",
# pa se pretvara u has_company/has_agent, a ne briše.
def plot_missing_values(raw: pd.DataFrame) -> None:
    """01: which columns have missing values in the raw file, and how many."""
    missing_count = raw.isna().sum()
    missing_count = missing_count[missing_count > 0]
    missing_pct = (missing_count / len(raw) * 100).sort_values()

    fig, ax = plt.subplots(figsize=(8, 3.5))
    missing_pct.plot(kind="barh", ax=ax, color=PURPLE)
    for bar, column in zip(ax.patches, missing_pct.index):
        ax.annotate(f" {num(missing_pct[column], 2)} % ({thousands(missing_count[column])})",
                    (bar.get_width(), bar.get_y() + bar.get_height() / 2), va="center", fontsize=9)
    ax.set_xlim(0, 110)
    ax.set_xlabel("Udio nedostajućih vrijednosti (%)")
    ax.set_ylabel("Kolona")
    ax.set_title(f"Nedostajuće vrijednosti u izvornom skupu "
                 f"({thousands(len(raw))} redova, {raw.shape[1]} kolona)")
    emit(fig, "eda01_nedostajuce_vrijednosti.png")


# eda02 – Ukupni udio otkazivanja
# Zašto: pokazuje ravnotežu klasa i šta postiže "glupi" model.
# Zaključak: otkazano je 37 % rezervacija. Model koji uvijek kaže "neće otkazati"
# bio bi tačan u 63 % slučajeva, pa tačnost sama nije dovoljna metrika.
def plot_overall_rate(df: pd.DataFrame) -> None:
    """02: how many bookings are cancelled overall (the class balance)."""
    shares = df[TARGET].value_counts(normalize=True).sort_index() * 100

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    shares.plot(kind="bar", ax=ax, color=[GREEN, RED])
    for bar, share in zip(ax.patches, shares):
        ax.annotate(f"{num(share, 1)} %", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom")
    ax.set_xticks([0, 1], STATUS_LABELS, rotation=0)
    ax.set_xlabel("Status rezervacije")
    ax.set_ylabel("Udio rezervacija (%)")
    ax.set_title("Raspodjela ciljne varijable is_canceled")
    emit(fig, "eda02_ukupni_udio_otkazivanja.png")


# eda03–eda07 – Udio otkazivanja po kategorijskom obilježju
# Zašto: kategorijske varijable (hotel, depozit, segment, kanal, tip gosta) ne
# ulaze u korelacionu mapu, pa se njihov uticaj vidi ovdje.
# Zaključci su u komentarima uz svaki poziv u run_hotel().
def plot_rate_by_category(df: pd.DataFrame, column: str, filename: str, rotate: int = 0) -> None:
    """03, 04, 05, 06, 07: cancellation rate for each value of one category,
    highest first."""
    stats = cancellation_rate(df, column).sort_values("rate", ascending=False)
    stats = stats[stats["n"] >= MIN_GROUP_SIZE]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    rate_bars(ax, stats)
    ax.set_xlabel(COLUMN_LABELS.get(column, column))
    ax.set_title(f"Udio otkazivanja po obilježju {column}")
    plt.setp(ax.get_xticklabels(), rotation=rotate, ha="right" if rotate else "center",
             rotation_mode="anchor")
    emit(fig, filename)


# eda08 – Udio otkazivanja po mjesecu dolaska
# Zašto: provjera da li otkazivanje zavisi od sezone.
# Zaključak: najviše u proljeće i ljeto (apr–jun oko 41 %), najmanje zimi
# (jan i nov oko 31 %). Razlike postoje, ali su umjerene.
def plot_monthly_trend(df: pd.DataFrame) -> None:
    """08: cancellation rate per arrival month (all years together)."""
    monthly = cancellation_rate(df, "arrival_month_num")["rate"].sort_index()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(monthly.index, monthly.values, marker="o", color=BLUE)
    ax.set_xticks(range(1, 13), MONTHS_BS)
    ax.set_xlabel("Mjesec dolaska")
    ax.set_ylabel(CANCEL_AXIS)
    ax.set_title("Udio otkazivanja po mjesecu dolaska")
    emit(fig, "eda08_otkazivanje_po_mjesecu_dolaska.png")


# eda09 – Udio otkazivanja po vremenu najave
# Zašto: lead_time je numerička varijabla s najjačom vezom s otkazivanjem.
# Zaključak: što se ranije rezerviše, to se češće otkazuje – od 10 % (0–7 dana)
# do 68 % (više od godinu dana). Rast je stalan kroz sve grupe.
def plot_lead_time(df: pd.DataFrame) -> None:
    """09: cancellation rate per lead-time band -- how risky is a booking
    made X days ahead."""
    bands = pd.cut(df["lead_time"], LEAD_TIME_BINS, labels=LEAD_TIME_LABELS)
    stats = cancellation_rate(df, bands)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    rate_bars(ax, stats, show_rate=True)
    ax.grid(axis="x", visible=False)  # vertical grid lines would cut through the labels
    ax.set_ylim(0, stats["rate"].max() * 1.15)
    ax.set_xlabel("Vrijeme najave (dani od rezervacije do dolaska)")
    ax.set_title("Udio otkazivanja prema vremenu najave")
    emit(fig, "eda09_otkazivanje_po_vremenu_najave.png")


def correlation_heatmap(ax, corr: pd.DataFrame, **heatmap_options) -> None:
    """Heatmap of correlations, annotated with decimal-comma values."""
    sns.heatmap(corr, annot=corr.map(lambda v: num(v, 2)), fmt="", cmap="coolwarm", center=0,
                ax=ax, cbar_kws={"label": CORR_LABEL}, **heatmap_options)


# eda10 – Korelacija numeričkih obilježja s otkazivanjem
# Zašto: brz pregled koja numerička obilježja imaju linearnu vezu s ciljem.
# Zaključak: sve veze su slabe (najjača lead_time 0,29 i posebni zahtjevi
# −0,24), pa nijedno obilježje samo ne predviđa otkazivanje – potrebni su
# nelinearni modeli. room_type_changed i parking su jaki, ali curenje (eda14).
def plot_correlation_with_target(df: pd.DataFrame) -> None:
    """10: correlation of each numeric feature with is_canceled, sorted."""
    corr = df.select_dtypes(include="number").corr()[[TARGET]].drop(index=TARGET)
    corr = corr.sort_values(by=TARGET, ascending=False)

    fig, ax = plt.subplots(figsize=(4.5, 8))
    correlation_heatmap(ax, corr)
    ax.set_xticklabels([TARGET])
    ax.set_title("Korelacija numeričkih obilježja s is_canceled")
    emit(fig, "eda10_korelacija_s_otkazivanjem.png")


# eda11 – Korelaciona matrica
# Zašto: otkriva obilježja koja nose istu informaciju (multikolinearnost).
# Zaključak: jako povezani su samo parovi koji su izvedeni jedan iz drugog
# (mjesec–sedmica 0,99, noćenja radnim danima–ukupno 0,94, djeca–is_family
# 0,90). Stablima to ne smeta; kod logističke regresije to treba imati na umu.
def plot_correlation_matrix(df: pd.DataFrame) -> None:
    """11: correlations between every pair of numeric features."""
    corr = df.select_dtypes(include="number").corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    correlation_heatmap(ax, corr, vmin=-1, vmax=1, annot_kws={"size": 6})
    ax.set_title("Korelaciona matrica numeričkih obilježja")
    emit(fig, "eda11_korelaciona_matrica.png")


# eda12 – Broj rezervacija po mjesecu (jul 2015 – avg 2017)
# Zašto: prikazuje potražnju kroz vrijeme – osnova za prognozu potražnje.
# Zaključak: jasna godišnja sezonalnost – najviše dolazaka u proljeće i jesen
# (apr–maj, sep–okt), manje u julu i avgustu, najmanje zimi. Pazi: jul i avg
# se u podacima javljaju 3 godine, ostali mjeseci 2, pa zbirovi po mjesecu
# bez dijeljenja brojem godina lažno ističu ljeto.
def plot_monthly_bookings(df: pd.DataFrame) -> None:
    """12: bookings per arrival month, stacked: realized at the bottom,
    cancelled on top."""
    month = pd.to_datetime(dict(year=df["arrival_date_year"], month=df["arrival_month_num"], day=1))
    counts = df.groupby([month, TARGET]).size().unstack(fill_value=0).sort_index()
    realized, cancelled = counts[0], counts[1]

    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = np.arange(len(counts))
    ax.bar(x, realized, color=GREEN, label="Realizovane")
    ax.bar(x, cancelled, bottom=realized, color=RED, label="Otkazane")
    ax.set_xticks(x, [f"{MONTHS_BS[d.month - 1]} {d.year}" for d in counts.index])
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right", rotation_mode="anchor")
    ax.set_xlabel("Mjesec dolaska")
    ax.set_ylabel("Broj rezervacija")
    ax.set_title("Broj rezervacija po mjesecu dolaska – ukupna i realizovana potražnja")
    ax.legend()
    emit(fig, "eda12_rezervacije_po_mjesecu.png")


# eda13 – Posebni zahtjevi i ranija otkazivanja
# Zašto: dva obilježja ponašanja gosta koja ukazuju na ozbiljnost namjere.
# Zaključak: bez posebnih zahtjeva otkazuje se 48 %, s jednim ili više oko
# 10–22 %. Gost koji je ranije otkazao otkazuje opet u 53–95 % slučajeva.
def plot_requests_and_history(df: pd.DataFrame) -> None:
    """13: cancellation rate by number of special requests (left) and by
    number of earlier cancellations (right)."""
    earlier = pd.cut(df["previous_cancellations"], [-1, 0, 1, np.inf], labels=["0", "1", "2 i više"])
    panels = [
        (df["total_of_special_requests"], BLUE, "Broj posebnih zahtjeva", "Posebni zahtjevi gosta"),
        (earlier, ORANGE, "Broj ranijih otkazivanja gosta", "Historija otkazivanja"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (groups, color, xlabel, title) in zip(axes, panels):
        rate_bars(ax, cancellation_rate(df, groups), color=color)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
    fig.suptitle("Udio otkazivanja prema ponašanju gosta")
    emit(fig, "eda13_otkazivanje_po_zahtjevima_i_historiji.png")


# eda14 – Obilježja koja nastaju tek pri dolasku gosta
# Zašto: dokaz da ova dva obilježja "znaju" ishod i moraju van iz modela.
# Zaključak: nijedna od 7.393 rezervacije s parkingom nije otkazana, a s
# promijenjenom sobom samo 5,5 %. Soba i parking se upisuju kad gost stigne,
# pa bi model s njima varao (curenje podataka).
def plot_post_booking_features(df: pd.DataFrame) -> None:
    """14: room change and parking request. Both are filled in only when the
    guest arrives, so a cancelled booking almost never has them -- the
    evidence for excluding them from the models."""
    room = df["room_type_changed"].map({0: "Ista kao rezervisana", 1: "Različita"})
    parking = pd.Series(np.where(df["required_car_parking_spaces"] > 0, "Traženo", "Nije traženo"),
                        index=df.index)
    panels = [
        (room, "Dodijeljena soba", ["Ista kao rezervisana", "Različita"]),
        (parking, "Parking mjesto", ["Nije traženo", "Traženo"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (groups, xlabel, order) in zip(axes, panels):
        rate_bars(ax, cancellation_rate(df, groups).reindex(order), color=[BLUE, RED], show_rate=True)
        ax.set_ylim(0, 50)
        ax.set_xlabel(xlabel)
    fig.suptitle("Obilježja koja se popunjavaju tek pri dolasku gosta (isključena iz modela)")
    emit(fig, "eda14_obiljezja_s_dolaska.png")


def run_hotel() -> None:
    raw = load_data(HOTEL_DATA)
    df = load_and_prepare(HOTEL_DATA)
    print(f"Hotel: {len(raw)} raw rows, {len(df)} after cleaning.")

    plot_missing_values(raw)
    plot_overall_rate(df)
    # eda03: City Hotel otkazuje češće (42 %) od Resort Hotela (28 %).
    plot_rate_by_category(df, "hotel", "eda03_otkazivanje_po_hotelu.png")
    # eda04: Non Refund 99 % – grupne rezervacije agencija, uglavnom duplikati.
    plot_rate_by_category(df, "deposit_type", "eda04_otkazivanje_po_vrsti_depozita.png")
    # eda05: grupe (61 %) i agencije (35–37 %) otkazuju najviše, direktni i korporativni najmanje.
    plot_rate_by_category(df, "market_segment", "eda05_otkazivanje_po_trzisnom_segmentu.png", rotate=30)
    # eda06: preko agencija (TA/TO) ide 82 % rezervacija i one otkazuju 41 %, direktne 18 %.
    plot_rate_by_category(df, "distribution_channel", "eda06_otkazivanje_po_kanalu_distribucije.png")
    # eda07: pojedinačni gosti (Transient) otkazuju 41 %, grupe s ugovorom (Group) samo 10 %.
    plot_rate_by_category(df, "customer_type", "eda07_otkazivanje_po_tipu_gosta.png")
    plot_monthly_trend(df)
    plot_lead_time(df)
    plot_correlation_with_target(df)
    plot_correlation_matrix(df)
    plot_monthly_bookings(df)
    plot_requests_and_history(df)
    plot_post_booking_features(df)


# =============================================================================
# Credit card transactions
# =============================================================================

# eda15 – Iznos transakcije: legitimne prema prevarama
# Zašto: Amount je jedno od rijetkih neanonimiziranih obilježja kartica.
# Zaključak: prevare su češće vrlo male (medijan 9,25 prema 22 EUR – testiranje
# ukradene kartice), ali i prosječno veće (122 prema 88 EUR). Raspodjele se
# jako preklapaju, pa iznos sam ne otkriva prevaru.
def plot_card_amount(df: pd.DataFrame) -> None:
    """15: transaction amounts, legitimate vs. fraud. The log scale stops a
    few very large payments from squeezing everything else into one bar."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for cls, label, color in [(0, "Legitimne", GREEN), (1, "Prevare", RED)]:
        amounts = df.loc[df["Class"] == cls, "Amount"]
        ax.hist(np.log1p(amounts), bins=50, density=True, alpha=0.6, label=label, color=color)
    ax.set_xlabel("log(1 + iznos transakcije u EUR)")
    ax.set_ylabel("Gustina")
    ax.set_title("Raspodjela iznosa transakcija po klasi")
    ax.legend()
    emit(fig, "eda15_kartice_iznos_po_klasi.png")


# eda16 – Udio prevara po satu
# Zašto: provjera da li se prevare dešavaju u određeno doba dana.
# Zaključak: noću (sati 0–6 od početka snimanja) ima malo transakcija (8 %),
# ali četvrtina svih prevara – udio prevara je tada i do 10 puta iznad prosjeka.
def plot_card_fraud_by_hour(df: pd.DataFrame) -> None:
    """16: share of fraud per hour of the day. 'Time' counts seconds from the
    first transaction, so the hour is only relative to that start."""
    hour = (df["Time"] // 3600) % 24
    fraud_rate = df.groupby(hour)["Class"].mean() * 100
    average = df["Class"].mean() * 100

    fig, ax = plt.subplots(figsize=(10, 4.5))
    fraud_rate.plot(kind="bar", ax=ax, color=RED, rot=0)
    ax.axhline(average, color="gray", linestyle="--", label=f"Prosjek ({num(average, 3)} %)")
    ax.set_xlabel("Sat u danu")
    ax.set_ylabel("Udio prevarnih transakcija (%)")
    ax.set_title("Udio prevara po satu u danu")
    ax.legend()
    emit(fig, "eda16_kartice_prevare_po_satu.png")


def run_card() -> None:
    if not CARD_DATA.exists():
        print(f"Skipping card EDA -- {CARD_DATA} not found.")
        return
    df = pd.read_csv(CARD_DATA)
    print(f"Card: {len(df)} transactions, {int(df['Class'].sum())} fraudulent.")

    plot_card_amount(df)
    plot_card_fraud_by_hour(df)


# =============================================================================
# Entry point
# =============================================================================

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
