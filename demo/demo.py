"""Kratki demo za odbranu (oko 1 minute izvršavanja).

1. Model za otkazivanje (XGBoost) ocjenjuje tri nove rezervacije.
2. Model za prevaru (Random Forest) provjerava transakcije iz testnog skupa,
   koje model nikad nije vidio tokom treniranja.

Pokretanje:  python demo/demo.py
"""

import sys
import tempfile
import time
import contextlib
import io
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from card_fraud import load_modeling_data as load_card  # noqa: E402
from data_prep import get_feature_lists, load_and_prepare  # noqa: E402
from modeling import positive_scores, split_data  # noqa: E402


def naslov(t):
    print("\n" + "=" * 70 + f"\n  {t}\n" + "=" * 70)


def pauza():
    time.sleep(1.2)


# ----------------------------------------------------------------- 1. otkazivanje
naslov("1. PREDVIĐANJE OTKAZIVANJA REZERVACIJE  (model: XGBoost)")

template = pd.read_csv(ROOT / "data" / "raw" / "hotel_bookings.csv", nrows=1)
template = template.drop(columns=["is_canceled", "reservation_status", "reservation_status_date"])
rezervacije = {
    "A – grupna preko agencije, 300 dana unaprijed, ranije otkazivala": dict(
        lead_time=300, country="PRT", market_segment="Groups", previous_cancellations=1,
        customer_type="Transient-Party", adr=85.0),
    "B – online agencija, 90 dana unaprijed, bez posebnih zahtjeva": dict(
        lead_time=90, country="FRA"),
    "C – direktna, 5 dana unaprijed, 2 posebna zahtjeva": dict(
        hotel="Resort Hotel", lead_time=5, country="GBR", market_segment="Direct",
        distribution_channel="Direct", agent=None, total_of_special_requests=2,
        is_repeated_guest=1, adr=120.0),
}
# zajednička polja: City Hotel, maj 2017., 4 noći, online agencija
zajednicko = dict(
    hotel="City Hotel", market_segment="Online TA", distribution_channel="TA/TO", agent=9,
    booking_changes=0, stays_in_weekend_nights=1, stays_in_week_nights=3,
    arrival_date_year=2017, arrival_date_month="May", arrival_date_week_number=20,
    adr=110.0, total_of_special_requests=0, reserved_room_type="A", assigned_room_type="A")
rows = []
for opis, izmjene in rezervacije.items():
    r = template.iloc[0].copy()
    for k, v in {**zajednicko, **izmjene}.items():
        r[k] = v
    rows.append(r)
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "nove_rezervacije.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    df = load_and_prepare(path, require_target=False)

model = joblib.load(ROOT / "models" / "best_model.joblib")
num, cat = get_feature_lists(df)
proba = model.predict_proba(df[num + cat])[:, 1]
for opis, p in zip(rezervacije, proba):
    pauza()
    nivo = "VISOK" if p >= 0.6 else ("SREDNJI" if p >= 0.3 else "NIZAK")
    print(f"\n  Rezervacija {opis}")
    print(f"     vjerovatnoća otkazivanja: {f"{p * 100:5.1f}".replace(".", ",")} %   →  rizik: {nivo}")

# ----------------------------------------------------------------- 2. prevara
pauza()
naslov("2. DETEKCIJA PREVARE  (model: Random Forest)")
print("  Učitavam transakcije i izdvajam testni skup (model ga nije vidio)...")
with contextlib.redirect_stdout(io.StringIO()):
    X, y = load_card(ROOT / "data" / "raw" / "creditcard.csv")
_, X_test, _, y_test = split_data(X, y)
saved = joblib.load(ROOT / "models" / "card_fraud_model.joblib")
pipe, prag = saved["pipeline"], saved["threshold"]

rng = np.random.RandomState(7)
idx = np.concatenate([rng.choice(np.flatnonzero(y_test == 1), 4, replace=False),
                      rng.choice(np.flatnonzero(y_test == 0), 4, replace=False)])
rng.shuffle(idx)
sample = X_test.iloc[idx]
scores = positive_scores(pipe, sample)
print("\n  Prag odlučivanja: " + f"{prag:.2f}".replace(".", ",") + "\n")
print(f"  {'Iznos':>10}  {'Ocjena':>7}  {'Odluka modela':<16} {'Stvarno':<12}")
for (_, row), s, true in zip(sample.iterrows(), scores, y_test.iloc[idx]):
    pauza()
    odluka = "BLOKIRAJ" if s >= prag else "odobri"
    stvarno = "prevara" if true == 1 else "legitimna"
    tacno = "✓" if (s >= prag) == (true == 1) else "✗ greška"
    iznos = f"{np.expm1(row['log_amount']):>8.2f}".replace(".", ",")
    ocjena = f"{s:>7.2f}".replace(".", ",")
    print(f"  {iznos} €  {ocjena}  {odluka:<16} {stvarno:<12} {tacno}")

all_scores = positive_scores(pipe, X_test)
pred = all_scores >= prag
pauza()
print(f"\n  Cijeli testni skup ({len(y_test):,} transakcija):".replace(",", "."))
print(f"     uhvaćeno prevara: {int((pred & (y_test == 1)).sum())} od {int(y_test.sum())}")
print(f"     lažne uzbune:     {int((pred & (y_test == 0)).sum())}")
print()
