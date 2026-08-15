# Predviđanje potražnje i detekcija prevare

Projekat mašinskog učenja iz dva dijela.

**Dio 1 — Potražnja:** prognoza ukupne hotelske potražnje kroz vrijeme i predviđanje
rizika otkazivanja za pojedinačnu rezervaciju. Ta dva zadatka su povezana:
`neto potražnja = bruto rezervacije − otkazivanja`.

**Dio 2 — Prevara:** detekcija prevarnih transakcija, i u *nadziranom* okruženju
(označeni podaci o kreditnim karticama) i u *nenadziranom* okruženju (hotelske
rezervacije, gdje oznaka prevare ne postoji).

---

## 1. Skupovi podataka

| Skup podataka | Redova | Ciljna varijabla | Izvor |
|---|---|---|---|
| `hotel_bookings.csv` | 119.390 (118.565 nakon čišćenja) | `is_canceled` (37,3 % pozitivnih) | Antonio, Almeida i Nunes (2019) |
| `creditcard.csv` | 284.807 | `Class` (0,172 % prevara) | ULB Machine Learning Group |

Oba se očekuju u direktoriju `data/raw/`. Nijedan nije uključen u repozitorij.

- Hotel: <https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand>
- Kartice: <https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>

---

## 2. Struktura projekta

```
├── data/raw/                  # ulazni CSV fajlovi (nisu u repozitoriju)
├── src/
│   ├── data_prep.py           # zajedničko učitavanje, čišćenje, inženjering obilježja
│   ├── eda.py                 # eksplorativna analiza → grafici
│   ├── train.py               # klasifikatori otkazivanja (LR / RF / XGBoost)
│   ├── predict.py             # ocjenjivanje novih rezervacija sačuvanim modelom
│   ├── demand_forecast.py     # prognoza potražnje (naivni / SARIMA / GBM)
│   ├── fraud_detection.py     # nenadzirana trijaža prevara na hotelskim podacima
│   ├── card_fraud.py          # nadzirana klasifikacija prevara na karticama
│   ├── tuning.py              # RandomizedSearchCV, ponovljena validacija, referentni modeli
│   ├── explain.py             # SHAP atribucija + pragovi osjetljivi na trošak
│   └── plotting.py            # zajednički pomoćni modul za prikaz/čuvanje grafika
├── tests/                     # pytest testovi (56 testova)
├── models/                    # serijalizovani modeli (.joblib)
├── reports/
│   ├── figures/               # svi grafici (fiksna imena, vidjeti §4.2)
│   └── *.csv                  # tabele rezultata
└── requirements.txt
```

---

## 3. Postavljanje

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Zatim smjestite `hotel_bookings.csv` i `creditcard.csv` u `data/raw/`.

## 4. Pokretanje

```bash
python src/eda.py                    # eksplorativni grafici
python src/train.py                  # treniranje modela otkazivanja
python src/demand_forecast.py        # povratno testiranje prognoze potražnje
python src/fraud_detection.py        # nenadzirana trijaža hotelskih prevara
python src/card_fraud.py             # nadzirani model prevara na karticama
python src/tuning.py --task both     # pretraga hiperparametara + referentni modeli
python src/explain.py --task both    # SHAP + analiza troška
python src/predict.py --input-path data/raw/hotel_bookings.csv
```

### 4.1 Opcije komandne linije

| Opcija | Skripta | Zadana vrijednost | Šta radi |
|---|---|---|---|
| `--show` | sve koje crtaju grafike | isključeno | Otvara svaki grafik u prozoru čim nastane. Izvršavanje se pauzira dok ne zatvorite prozor. |
| `--data-path PUT` | `train.py`, `card_fraud.py`, `fraud_detection.py`, `demand_forecast.py` | `data/raw/...` | Učitava drugi ulazni CSV umjesto zadanog. |
| `--input-path PUT` | `predict.py` | **obavezno** | CSV s rezervacijama koje treba ocijeniti. Nema zadanu vrijednost jer skripta bez ulaznih podataka nema šta raditi. |
| `--output-path PUT` | `predict.py` | `reports/predictions.csv` | Gdje zapisati predviđanja. |
| `--model-path PUT` | `predict.py` | `models/best_model.joblib` | Koji sačuvani model koristiti. |
| `--use-smote` | `card_fraud.py` | isključeno | Umjesto težina klasa (`class_weight` / `scale_pos_weight`) koristi SMOTE naduzorkovanje manjinske klase. Korisno za poređenje dva pristupa neuravnoteženosti. |
| `--hotel "City Hotel"` | `demand_forecast.py` | oba hotela | Prognozira potražnju samo za jedan hotel. Prihvata `"City Hotel"` ili `"Resort Hotel"`. Bez ove opcije oba se sabiraju u jednu seriju. |
| `--test-days N` | `demand_forecast.py` | `90` | Koliko posljednjih dana izdvojiti za testiranje. Veći broj = pouzdanija procjena, ali manje podataka za treniranje. |
| `--review-cost IZNOS` | `explain.py` | `3.0` | Pretpostavljeni trošak (u €) ručnog pregleda jedne označene transakcije. Direktno pomjera troškovno optimalan prag: skuplji pregled → viši prag → manje lažnih uzbuna. |
| `--task {hotel,card,both}` | `tuning.py`, `explain.py` | `both` | Ograničava izvršavanje na jedan od dva zadatka. |
| `--n-iter N` | `tuning.py` | `15` | Broj nasumičnih kombinacija hiperparametara koje `RandomizedSearchCV` isprobava. Veći broj = bolji rezultat, ali linearno duže izvršavanje. |
| `--cv-folds N` | `tuning.py` | `3` | Broj podskupova unakrsne validacije tokom pretrage. |
| `--n-repeats N` | `tuning.py` | `2` | Broj ponavljanja 5-struke validacije za interval pouzdanosti (§5.7). |
| `--top-n N` | `fraud_detection.py` | `500` | Koliko najrizičnijih rezervacija zapisati u red za pregled. |

Primjeri:

```bash
python src/demand_forecast.py --hotel "City Hotel" --test-days 120
python src/card_fraud.py --use-smote
python src/explain.py --task card --review-cost 5.0
python src/tuning.py --task hotel --n-iter 40
```

### 4.2 Grafici

Svaka skripta koja crta grafike prihvata opciju `--show`, koja svaki grafik otvara
u prozoru čim je napravljen (izvršavanje se pauzira na svakom grafiku dok se
prozor ne zatvori):

```bash
python src/eda.py --show
```

`SHOW_FIGURES=1` je ekvivalent putem varijable okruženja. Bez ijedne od te dvije
opcije skripte se izvršavaju bez prikaza.

**Svi grafici se čuvaju** u `reports/figures/`. Zapisivanje je centralizovano u
`src/plotting.py`, gdje svaki grafik ima fiksno ime fajla (npr.
`07_roc_curves.png`). Zbog toga ponovno pokretanje skripte **prepisuje** njen
prethodni izlaz na istom mjestu umjesto da dodaje nove kopije — u direktoriju
uvijek stoji tačno jedna, aktuelna verzija svakog grafika, bez obzira na to
koliko puta ste skriptu pokrenuli.

---

## 5. Metodologija i rezultati

### 5.1 Prognoza potražnje

Pojedinačne rezervacije agregirane su u dnevnu seriju dolazaka (793 dana,
01.07.2015. → 31.08.2017.; prosječno 149,5 bruto / 93,8 neto rezervacija dnevno).

Evaluacija koristi **vremensku podjelu** (posljednjih 90 dana izdvojeno) i
**povratno testiranje s prognozom za jedan korak unaprijed**. Nasumična podjela
bi propustila informacije iz budućnosti u treniranje i nikada nije validna za
vremenske serije.

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| **SARIMA(1,1,1)(1,1,1,7)** | **15,02** | **18,70** | **14,9 %** |
| Sezonski naivni (t−7) | 19,59 | 24,14 | 19,1 % |
| Gradient boosting (obilježja s pomakom) | 19,73 | 24,30 | 19,9 % |
| Naivni (t−1) | 22,58 | 28,38 | 22,2 % |

SARIMA poboljšava sezonski naivni referentni model za **23,3 % MAE**.

> **Negativan rezultat vrijedan prijavljivanja:** gradient boosting *nije uspio
> nadmašiti sezonski naivni referentni model*. Na kratkoj dnevnoj seriji kojom
> dominira sedmična sezonalnost, klasični statistički model pobjeđuje. Prijaviti
> ovo je vrjednije nego tiho izostaviti poređenje.

### 5.2 Predviđanje otkazivanja

Korištena su isključivo obilježja poznata u trenutku rezervacije. Kategorijske
varijable su one-hot kodirane, numeričke standardizovane, sve unutar objekta
`Pipeline` kako bi se pretprocesiranje prilagođavalo samo na skupu za treniranje.

| Model | Tačnost | Preciznost | Odziv | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **XGBoost** | **0,879** | 0,856 | 0,810 | 0,833 | **0,952** |
| Random Forest | 0,867 | 0,804 | 0,851 | 0,827 | 0,947 |
| Logistička regresija | 0,818 | 0,738 | 0,792 | 0,764 | 0,901 |

**Najvažnija SHAP obilježja:** `country_PRT` (0,93), `deposit_type_Non Refund`
(0,88), `required_car_parking_spaces` (0,55), `total_of_special_requests` (0,54),
`lead_time` (0,53).

Rezervacije iz Portugala i nepovratni depoziti dominiraju odlukom modela — pri
čemu drugo, suprotno intuiciji, *povećava* predviđenu vjerovatnoću otkazivanja
u ovom skupu podataka, što je poznata osobenost vrijedna diskusije.

### 5.3 Curenje podataka — metodološka napomena

Široko rasprostranjeni notebook-ovi na ovom skupu podataka prijavljuju tačnost
od oko 99,5 %. Ta brojka je posljedica **curenja ciljne varijable**: kolone
`reservation_status` i `reservation_status_date` bilježe *ishod* rezervacije
(`Canceled` / `Check-Out` / `No-Show`) i time direktno kodiraju ciljnu varijablu.

`data_prep.py` uklanja obje (`LEAKAGE_COLS`). Pošten rezultat je **87,9 %
tačnosti / 0,952 ROC-AUC**. Svaki model koji na ovom skupu postiže gotovo savršen
rezultat treba provjeriti upravo zbog ovog problema.

### 5.4 Nadzirana detekcija prevara (podaci o karticama)

Uz 0,172 % pozitivnih primjera, **tačnost nije upotrebljiva metrika** —
predviđanje „nikad prevara” postiže 99,83 %. Primarna metrika je stoga
**prosječna preciznost (PR-AUC)**, a prag odlučivanja se eksplicitno podešava
umjesto da ostane na 0,5.

> ### Glavni rezultat
> **XGBoost, PR-AUC = 0,850 ± 0,029** (ponovljena stratifikovana 5-struka
> unakrsna validacija × 2, §5.7).
>
> Taj interval — a ne bilo koji pojedinačni broj — jeste rezultat koji treba
> navoditi. Postoji samo oko 394 prevara u skupu za treniranje, pa je varijansa
> među podskupovima značajna i jedna jedina podjela na trening/test ne može
> razlučiti razlike manje od ≈ 0,03.

Brojke iz jedne podjele prikazane u nastavku navode se zbog detalja matrice
konfuzije koji pružaju (koliko je prevara zaista uhvaćeno pri odabranom pragu),
a **ne** kao glavne tvrdnje o tačnosti:

| Model | PR-AUC *(jedna podjela)* | ROC-AUC | Preciznost | Odziv | Uhvaćeno / Propušteno | Lažne uzbune |
|---|---|---|---|---|---|---|
| **XGBoost** | 0,881 | 0,985 | 0,942 | 0,827 | 81 / 17 | 5 |
| Random Forest | 0,850 | 0,983 | 0,886 | 0,796 | 78 / 20 | 10 |
| Logistička regresija | 0,713 | 0,971 | 0,833 | 0,816 | 80 / 18 | 16 |

Vrijednost 0,881 za XGBoost nalazi se unutar intervala 0,850 ± 0,029 i predstavlja
optimističan uzorak iz njega; ne treba je pisati kao „model postiže 0,881”.
Razlika između XGBoost-a (0,881) i Random Forest-a (0,850) na toj jednoj podjeli
**manja je od jedne standardne devijacije** i stoga nije dokaz da jedan model
zaista nadmašuje drugi. Vrijednost logističke regresije (0,713) jasno je izvan
intervala, pa je ta razlika stvarna.

ROC-AUC iznosi 0,97–0,99 za *sva tri* modela i jedva ih razdvaja; PR-AUC ih
razdvaja ispravno (0,71 → 0,88). Zbog toga se PR-AUC preferira pri ekstremnoj
neuravnoteženosti klasa.

Neuravnoteženost klasa rješava se pomoću `scale_pos_weight` /
`class_weight="balanced"`, uz SMOTE dostupan kroz `--use-smote`. SMOTE se
primjenjuje **unutar** pipeline-a, tako da uvijek naduzorkuje isključivo
podskupove za treniranje.

**Najvažnija SHAP obilježja:** V14 (2,47), V4 (2,14), V12 (1,06), V10 (1,03) —
u skladu s objavljenom literaturom o ovom skupu podataka. Obilježja su
anonimizovana metodom PCA, pa semantička interpretacija nije moguća.

### 5.5 Odabir praga osjetljiv na trošak

Maksimizacija F1 mjere implicitno pretpostavlja da su lažno pozitivan i lažno
negativan nalaz jednako skupi. Nisu. Ako se propuštenoj prevari pripiše
**stvarni iznos transakcije**, a lažnoj uzbuni fiksni **trošak ručnog pregleda
od 3 €**:

| Strategija | Prag | Propušteno | Lažne uzbune | Ukupni trošak |
|---|---|---|---|---|
| Bez modela | — | 98 | 0 | 10.644,93 € |
| F1-optimalan | 0,9509 | 17 | 5 | 2.162,66 € |
| **Troškovno optimalan** | **0,0460** | **12** | **26** | **1.907,72 €** |

Ekonomski optimalan prag je **oko 20× niži** od F1-optimalnog: isplati se
prihvatiti pet puta više lažnih uzbuna da bi se uhvatilo pet prevara više.
Model smanjuje izloženost prevarama za **82,1 %**.

### 5.6 Nenadzirana trijaža prevara (hotelski podaci)

`hotel_bookings.csv` **nema oznaku prevare**, pa nadzirani model nije moguć niti
se može prijaviti ikakva tačnost. `fraud_detection.py` umjesto toga kombinuje
šest domenskih heuristika s algoritmom Isolation Forest, spojenih u ocjenu
rizika od 0 do 100 (60 % pravila / 40 % anomalija).

| Pravilo | Označeno | % |
|---|---|---|
| Skupa rezervacija u posljednji čas, bez depozita | 433 | 0,37 % |
| Sumnjiva nulta cijena | 292 | 0,25 % |
| Prekomjerne izmjene rezervacije (≥5) | 245 | 0,21 % |
| Serijski otkazivač | 154 | 0,13 % |
| Dug boravak bez osiguranja | 100 | 0,08 % |
| Nesklad grupne cijene | 67 | 0,06 % |

1.256 rezervacija (1,06 %) aktivira barem jedno pravilo; Isolation Forest je
označio 2.372 (2,00 %). Izlaz je rangiran **red za pregled od strane ljudskog
analitičara**, a ne presuda o prevari.

### 5.7 Referentni modeli, pretraga hiperparametara i intervali pouzdanosti

Prostor pretrage (`tuning.py`, identičan za oba zadatka, 15 kandidata × 3-struka
unakrsna validacija):

| Parametar | Raspodjela |
|---|---|
| `n_estimators` | randint(200, 700) |
| `max_depth` | randint(3, 10) |
| `learning_rate` | loguniform(0.01, 0.3) |
| `subsample` | uniform(0.6, 1.0) |
| `colsample_bytree` | uniform(0.6, 1.0) |
| `min_child_weight` | randint(1, 10) |

**Trivijalni referentni modeli** (`DummyClassifier`) postavljaju donju granicu:

| Zadatak | Referentni model (većinski / stratifikovani / prior) | Podešeni XGBoost |
|---|---|---|
| Otkazivanje rezervacija (ROC-AUC) | 0,500 / 0,499 / 0,500 | **0,9517 ± 0,0017** |
| Prevare na karticama (PR-AUC) | 0,00173 (= učestalost prevara) | **0,8504 ± 0,0292** |

Model za prevare na karticama je otprilike **490× bolji od slučajnog pogađanja**,
što je ispravan način izražavanja PR-AUC vrijednosti — za razliku od ROC-AUC,
njegova referentna vrijednost je učestalost pozitivne klase, a ne 0,5.

Najbolji pronađeni hiperparametri — hotel: `max_depth=6, learning_rate=0.220,
n_estimators=559, subsample=0.902, colsample_bytree=0.614, min_child_weight=2`;
kartice: `max_depth=4, learning_rate=0.080, n_estimators=563, subsample=0.806,
colsample_bytree=0.716, min_child_weight=3`.

> **Dva poštena zapažanja.**
> (1) Podešavanje hiperparametara nije donijelo praktično **ništa** na hotelskom
> zadatku: 0,9517 ± 0,0017 nakon podešavanja naspram 0,952 s ručno odabranim
> vrijednostima. Signal u ovom skupu podataka nosi kvalitet obilježja, a ne
> hiperparametri — rezultat koji vrijedi navesti, a ne sakriti.
> (2) Standardna devijacija kod prevara na karticama (**± 0,0292**) je za red
> veličine veća od hotelske (± 0,0017), jer postoji samo oko 394 prevara u skupu
> za treniranje. Vrijednost iz jedne podjele (0,881) iz §5.4 udobno se smješta
> unutar tog intervala, pa je **ne** treba prijavljivati kao preciznu. Upravo
> zbog toga je ponovljena unakrsna validacija neophodna kod problema s rijetkim
> događajima.

---

## 6. Ograničenja

- **Statistike pretprocesiranja računaju se na cijelom skupu podataka prije
  podjele.** Odsijecanje netipičnih vrijednosti `adr` (99,9. percentil) i
  sažimanje rijetkih država (<1 % učestalosti → `"Other"`) u `data_prep.py`
  izvedeni su iz svih redova, uključujući i one kasnije izdvojene za testiranje.
  Radi se o blagoj kontaminaciji između skupa za treniranje i testiranje: obje
  su *distribucijske* statistike koje nikada ne dodiruju ciljnu varijablu, pa to
  nije curenje ciljne varijable i uticaj na prijavljene rezultate je zanemariv —
  ali strogo ispravan pipeline bi ih prilagođavao isključivo na skupu za
  treniranje, putem prilagođenih transformatora unutar objekta `Pipeline`.
  Navedeno je ovdje umjesto da bude prešućeno.
- Trijaža hotelskih prevara je **nevalidirana** — bez stvarnih oznaka, njena
  preciznost i odziv su po samoj prirodi problema nepoznati.
- Skup podataka o karticama obuhvata dva dana transakcija iz 2013. godine iz
  jedne regije; rezultati se ne moraju nužno prenijeti na druge periode ili
  platne sisteme.
- Analiza troška pretpostavlja fiksni trošak pregleda od 3 € i potpuni povrat
  spriječene prevare; stvarna ekonomija povrata sredstava je složenija.
- Serija potražnje obuhvata oko 2 godine, što ograničava otkrivanje godišnje
  sezonalnosti (uočena su samo dva ljetna vrhunca).
- SHAP vrijednosti na podacima o karticama nisu semantički interpretabilne jer
  su V1–V28 PCA komponente.

## 7. Testovi

```bash
pip install pytest
python -m pytest tests/ -q
```

56 testova pokriva svojstva koja bi, ako se naruše, tiho poništila rezultate:

- **Zaštita od curenja podataka** — kolone `reservation_status` se uklanjaju,
  ciljna varijabla se nikada ne pojavljuje među obilježjima, i nijedno numeričko
  obilježje nema korelaciju > 0,95 s ciljnom varijablom (okidač za slučaj da se
  curenje ponovo pojavi).
- **Zaštita od gledanja unaprijed** — za svako obilježje s pomakom i pokretnim
  prosjekom u `demand_forecast.py` provjerava se da odgovara ispravno pomjerenoj
  seriji, i nijedno obilježje ne smije reprodukovati `y[t]`.
- **Regresijski test tipova podataka** — provjerava se prepoznavanje kategorijskih
  kolona `hotel`, `country`, `deposit_type` i drugih. pandas ≥ 2 tekstualne
  kolone pohranjuje tipom `string` umjesto `object`, pa naivna provjera
  `dtype == object` tiho pogrešno klasifikuje svaku kategorijsku kolonu kao
  numeričku.
- **Invarijante povratne konverzije i aritmetike** — `expm1(log_amount)` vraća
  originalni iznos u eurima koji koristi analiza troška; `total_cost` jednak je
  zbiru svojih komponenti; povećanje praga nikada ne može smanjiti broj
  propuštenih prevara.

## 8. Reproducibilnost

Sve nasumične operacije koriste fiksno sjeme (`RANDOM_STATE = 42`). Podjele su
stratifikovane (klasifikacija) ili hronološke (vremenske serije). Tačne verzije
paketa navedene su u `requirements.txt`.

## 9. Literatura

- Antonio, N., de Almeida, A., & Nunes, L. (2019). *Hotel booking demand datasets.*
  Data in Brief, 22, 41–49.
- Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2015).
  *Calibrating probability with undersampling for unbalanced classification.* IEEE SSCI.
- Lundberg, S. M., & Lee, S.-I. (2017). *A unified approach to interpreting model
  predictions.* NeurIPS.
- Chawla, N. V. et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique.* JAIR.
