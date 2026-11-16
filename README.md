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
| `hotel_bookings.csv` | 119.390 → 118.563 nakon čišćenja → **85.166 bez duplikata** | `is_canceled` (37,3 % sirovo; **27,8 %** bez duplikata) | Antonio, Almeida i Nunes (2019) |
| `creditcard.csv` | 284.807 → **280.922 bez duplikata** | `Class` (492 → 473 prevare, ≈ 0,17 %) | ULB Machine Learning Group |

Oba se očekuju u direktoriju `data/raw/`. Nijedan nije uključen u repozitorij.

- Hotel: <https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand>
- Kartice: <https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>

**Duplikati.** Hotelski skup sadrži 31.994 identična reda. Ako ostanu, ista
rezervacija završi i u skupu za treniranje i u testnom skupu, pa model biva
ocijenjen na redovima koje je već vidio. Zato se prije podjele uklanjaju
duplikati po obilježjima modela + ciljnoj varijabli (isto za kartice). Uticaj je
izmjeren u §5.3.

---

## 2. Struktura projekta

```
├── data/raw/                  # ulazni CSV fajlovi (nisu u repozitoriju)
├── src/
│   ├── modeling.py            # zajednički protokol: 4 modela, prostori pretrage, CV, metrike
│   ├── data_prep.py           # učitavanje, čišćenje, inženjering obilježja, uklanjanje duplikata
│   ├── eda.py                 # eksplorativna analiza → grafici
│   ├── tuning.py              # RandomizedSearchCV za sva 4 modela, oba zadatka
│   ├── train.py               # klasifikatori otkazivanja (LR / SVM / RF / XGBoost)
│   ├── card_fraud.py          # nadzirana klasifikacija prevara na karticama (isti 4 modela)
│   ├── explain.py             # SHAP atribucija + prag osjetljiv na trošak
│   ├── sensitivity.py         # koliko duplikati, obilježja s dolaska i podjela mijenjaju rezultat
│   ├── imbalance.py           # bez korekcije vs. težine klasa vs. SMOTE
│   ├── demand_forecast.py     # prognoza potražnje (naivni / SARIMA / GBM)
│   ├── fraud_detection.py     # nenadzirana trijaža prevara na hotelskim podacima
│   ├── predict.py             # ocjenjivanje novih rezervacija sačuvanim modelom
│   ├── run_all.py             # pokreće sve navedeno redom
│   └── plotting.py            # zajedničko čuvanje grafika (300 dpi, decimalni zarez)
├── tests/                     # pytest testovi (98 testova)
├── models/                    # serijalizovani modeli (.joblib)
├── reports/
│   ├── figures/               # svi grafici (fiksna imena, vidjeti §4.2)
│   ├── *.csv                  # tabele rezultata
│   └── *_best_params.json     # najbolji hiperparametri iz tuning.py
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

Sve odjednom (≈ 55 minuta na 8-jezgrenom računaru, od čega je većina pretraga
hiperparametara):

```bash
python src/run_all.py                # cijela analiza
python src/run_all.py --skip-tuning  # koristi postojeće reports/*_best_params.json (≈ 20 min)
```

Ili korak po korak, ovim redom (`train.py`, `card_fraud.py` i `explain.py`
čitaju hiperparametre koje je zapisao `tuning.py`; bez tog fajla koriste zadane):

```bash
python src/eda.py
python src/tuning.py --task both
python src/train.py
python src/card_fraud.py
python src/explain.py --task both
python src/sensitivity.py
python src/imbalance.py
python src/demand_forecast.py
python src/fraud_detection.py
python src/predict.py --input-path data/raw/hotel_bookings.csv
```

### 4.1 Opcije komandne linije

| Opcija | Skripta | Zadana vrijednost | Šta radi |
|---|---|---|---|
| `--show` | sve koje crtaju grafike | isključeno | Otvara svaki grafik u prozoru čim nastane. Izvršavanje se pauzira dok ne zatvorite prozor. |
| `--data-path PUT` | `train.py`, `card_fraud.py`, `sensitivity.py`, `imbalance.py`, `fraud_detection.py`, `demand_forecast.py` | `data/raw/...` | Učitava drugi ulazni CSV umjesto zadanog. |
| `--task {hotel,card,both}` | `eda.py`, `tuning.py`, `explain.py` | `both` | Ograničava izvršavanje na jedan od dva zadatka. |
| `--models IME …` | `tuning.py` | sva 4 | Podešava samo navedene modele (`logistic_regression svm random_forest xgboost`). Ostali zadržavaju ranije sačuvane vrijednosti. |
| `--n-iter N` | `tuning.py` | `15` | Broj nasumičnih kombinacija hiperparametara po modelu. |
| `--cv-folds N` | `tuning.py` | `3` | Broj podskupova unakrsne validacije tokom pretrage. |
| `--n-repeats N` | `train.py`, `card_fraud.py` | `2` | Broj ponavljanja 5-struke unakrsne validacije. |
| `--review-cost IZNOS` | `explain.py` | `3.0` | Pretpostavljeni trošak (u €) ručnog pregleda jedne označene transakcije. Skuplji pregled → viši prag → manje lažnih uzbuna. |
| `--hotel "City Hotel"` | `demand_forecast.py` | oba hotela | Prognozira potražnju samo za jedan hotel. |
| `--test-days N` | `demand_forecast.py` | `90` | Koliko posljednjih dana izdvojiti za testiranje. |
| `--top-n N` | `fraud_detection.py` | `500` | Koliko najrizičnijih rezervacija zapisati u red za pregled. |
| `--input-path PUT` | `predict.py` | **obavezno** | CSV s rezervacijama koje treba ocijeniti. |
| `--output-path PUT` | `predict.py` | `reports/predictions.csv` | Gdje zapisati predviđanja. |
| `--model-path PUT` | `predict.py` | `models/best_model.joblib` | Koji sačuvani model koristiti. |
| `--skip-tuning` | `run_all.py` | isključeno | Preskače `tuning.py`. |

### 4.2 Grafici

Svi grafici se čuvaju u `reports/figures/` s fiksnim imenima, pa ponovno
pokretanje prepisuje stari izlaz umjesto da dodaje kopije. Natpisi su na
bosanskom, s decimalnim zarezom, u rezoluciji 300 dpi (spremno za rad).
`--show` ili `SHOW_FIGURES=1` ih prikazuje u prozoru.

| Broj | Sadržaj | Skripta |
|---|---|---|
| 01–06, 21–26, 31–32 | EDA hotelskog skupa (stopa otkazivanja po segmentima, nedostajuće vrijednosti, korelacije, obilježja s dolaska) | `eda.py` |
| 27–28 | EDA kartica (iznos po klasi, udio prevara po satu) | `eda.py` |
| 07–09 | ROC krive, važnost obilježja, matrica konfuzije — otkazivanje | `train.py` |
| 12, 13, 29 | PR krive, kompromis praga, matrica konfuzije — kartice | `card_fraud.py` |
| 18–20 | SHAP (hotel, kartice), kriva troška | `explain.py` |
| 30 | Studija osjetljivosti | `sensitivity.py` |
| 33 | Poređenje tretmana neuravnoteženosti | `imbalance.py` |
| 14–17 | Serija potražnje, dekompozicija, prognoza, sezonalnost | `demand_forecast.py` |
| 10–11 | Raspodjela ocjene rizika, pogoci pravila | `fraud_detection.py` |

---

## 5. Metodologija i rezultati

### 5.0 Zajednički protokol

Oba klasifikacijska zadatka koriste isti postupak (`modeling.py`):

1. Uklanjanje duplikata, zatim stratifikovana podjela **80 % trening / 20 % test**
   (`RANDOM_STATE = 42`).
2. Četiri modela iz prijave teme: **logistička regresija, SVM (RBF jezgro),
   Random Forest, XGBoost**, plus `DummyClassifier` kao donja granica.
   Pretprocesiranje (standardizacija, one-hot kodiranje s grupisanjem kategorija
   rjeđih od 1 % u „ostale”) nalazi se unutar `Pipeline`-a, pa se uči samo na
   podacima za treniranje.
3. **Pretraga hiperparametara** (`RandomizedSearchCV`, 15 kombinacija × 3-struka
   CV) isključivo na skupu za treniranje. Prostor pretrage i pobjedničke
   vrijednosti: `reports/{hotel,card}_search_space.csv`.
4. **Ponovljena stratifikovana 5-struka CV × 2** na skupu za treniranje daje
   srednju vrijednost ± standardnu devijaciju svake metrike.
5. **Najbolji model se bira po CV rezultatu**, ne po testnom skupu. Testni skup
   se koristi samo jednom, za konačnu procjenu.
6. Pragovi odlučivanja (F1, trošak) biraju se na *out-of-fold* predviđanjima
   skupa za treniranje, nikada na testnom skupu.

SVM s RBF jezgrom ima kvadratnu složenost, pa se trenira na stratifikovanom
uzorku od 20.000 redova (kod kartica: sve prevare + nasumični legitimni redovi).
Ocjenjuje se ipak na svim redovima testnog skupa.

### 5.1 Prognoza potražnje

Pojedinačne rezervacije agregirane su u dnevnu seriju dolazaka (793 dana,
01.07.2015. → 31.08.2017.). Evaluacija koristi **vremensku podjelu**
(posljednjih 90 dana) i **povratno testiranje s prognozom za jedan korak
unaprijed**. Nasumična podjela bi propustila informacije iz budućnosti.

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| **SARIMA(1,1,1)(1,1,1,7)** | **15,02** | **18,70** | **14,9 %** |
| Gradient boosting (obilježja s pomakom) | 19,17 | 23,62 | 19,3 % |
| Sezonski naivni (t−7) | 19,59 | 24,14 | 19,1 % |
| Naivni (t−1) | 22,58 | 28,38 | 22,2 % |

SARIMA poboljšava sezonski naivni referentni model za **23,3 % MAE**. Gradient
boosting je tek neznatno bolji od sezonskog naivnog modela po MAE (a lošiji po
MAPE): na kratkoj dnevnoj seriji kojom dominira sedmična sezonalnost klasični
statistički model jasno pobjeđuje.

### 5.2 Predviđanje otkazivanja

68.132 rezervacije za treniranje, 17.034 za testiranje (27,8 % otkazanih).
Korištena su samo obilježja poznata u trenutku rezervacije (§5.3).

| Model | CV ROC-AUC (± std) | Test ROC-AUC | PR-AUC | Tačnost | Preciznost | Odziv | F1 |
|---|---|---|---|---|---|---|---|
| Referentni (većinska klasa) | 0,500 | 0,500 | 0,278 | 0,722 | — | 0 | 0 |
| Logistička regresija | 0,810 ± 0,004 | 0,811 | 0,623 | 0,737 | 0,518 | 0,746 | 0,612 |
| SVM | 0,849 ± 0,003 | 0,852 | 0,710 | 0,811 | 0,721 | 0,521 | 0,605 |
| Random Forest | 0,885 ± 0,003 | 0,886 | 0,755 | 0,805 | 0,616 | **0,793** | **0,693** |
| **XGBoost** (izabran) | **0,889 ± 0,003** | **0,891** | **0,767** | **0,831** | **0,741** | 0,602 | 0,664 |

XGBoost je izabran po CV ROC-AUC. Razlika prema Random Forest-u (0,889 naspram
0,885) je približno jedna standardna devijacija, pa su dva ansambl modela
praktično izjednačena. Random Forest s `class_weight="balanced"` hvata više
otkazivanja (odziv 0,79) uz više lažnih uzbuna, što ima veći F1. Izbor između
njih zavisi od toga šta hotel više košta. Oba ansambla su jasno bolja od
linearnih modela (≈ 0,08 ROC-AUC iznad logističke regresije).

**Najvažnija SHAP obilježja (XGBoost):** `country = PRT` (0,70), `lead_time`
(0,54), `total_of_special_requests` (0,48), `market_segment = Online TA` (0,38),
`adr` (0,26).

### 5.3 Curenje podataka i studija osjetljivosti

Široko rasprostranjeni notebook-ovi na ovom skupu prijavljuju tačnost ≈ 99,5 %
jer koriste `reservation_status` / `reservation_status_date`, koje bilježe
*ishod* rezervacije. `data_prep.py` ih uklanja (`LEAKAGE_COLS`).

Dva manje očita izvora optimizma su također uklonjena:

- **Obilježja zabilježena tek pri dolasku gosta** (`POST_BOOKING_FEATURES`):
  `required_car_parking_spaces` (nijedna od 7.393 rezervacije s parkingom nije
  otkazana — parking se upisuje kad gost stigne) i `room_type_changed` (soba se
  mijenja pri prijavi: 5,5 % otkazivanja naspram 41,7 %). U trenutku rezervacije
  te informacije ne postoje.
- **Duplikati** (§1).

`sensitivity.py` mjeri koliko svaki od tih izbora mijenja rezultat (XGBoost,
isti hiperparametri, test ROC-AUC):

| Varijanta | ROC-AUC | PR-AUC | F1 |
|---|---|---|---|
| **1. Glavni model** (bez duplikata, bez obilježja s dolaska) | **0,891** | 0,767 | 0,664 |
| 2. + obilježja s dolaska | 0,909 | 0,794 | 0,691 |
| 3. Duplikati zadržani | 0,943 | 0,920 | 0,814 |
| 4. Duplikati + obilježja s dolaska (ranija verzija projekta) | 0,953 | 0,932 | 0,833 |
| 5. Bez obilježja `country` | 0,837 | 0,704 | 0,591 |
| 6. Bez `booking_changes` i `days_in_waiting_list` | 0,888 | 0,762 | 0,657 |
| 7. Vremenska podjela (trening 2015–2016, test 2017) | 0,783 | 0,625 | 0,508 |

Zaključci:

- Raniji rezultat 0,952 bio je **naduvan za ≈ 0,06 ROC-AUC**, uglavnom zbog
  duplikata (varijanta 3), a manjim dijelom zbog obilježja s dolaska.
- `country` nosi mnogo signala (−0,05 bez njega). Dio toga je vjerovatno
  specifičnost skupa (domaće portugalske rezervacije), što treba navesti.
- Na **vremenskoj podjeli** rezultat pada na 0,78: obrasci otkazivanja se mijenjaju
  iz godine u godinu. To je realnija procjena za stvarnu upotrebu (model se
  trenira na prošlosti, koristi u budućnosti).

### 5.4 Nadzirana detekcija prevara (podaci o karticama)

224.737 transakcija za treniranje (378 prevara), 56.185 za testiranje (95 prevara).
Uz ≈ 0,17 % pozitivnih, **tačnost nije upotrebljiva** (model „nikad prevara”
postiže 99,83 %). Primarna metrika je **PR-AUC**, čija je referentna vrijednost
učestalost prevara (0,0017), a ne 0,5.

| Model | CV PR-AUC (± std) | Test PR-AUC | Test ROC-AUC | Prag | Preciznost | Odziv | F1 | Uhvaćeno / propušteno | Lažne uzbune |
|---|---|---|---|---|---|---|---|---|---|
| Referentni | 0,002 | 0,002 | 0,500 | — | — | — | — | — | — |
| Logistička regresija | 0,752 ± 0,040 | 0,747 | 0,971 | 0,087 | 0,755 | 0,811 | 0,782 | 77 / 18 | 25 |
| SVM | 0,792 ± 0,049 | 0,792 | 0,945 | 0,296 | 0,705 | 0,832 | 0,763 | 79 / 16 | 33 |
| **Random Forest** (izabran) | **0,841 ± 0,026** | **0,842** | 0,959 | 0,440 | **0,906** | 0,811 | **0,856** | 77 / 18 | **8** |
| XGBoost | 0,835 ± 0,030 | 0,820 | 0,957 | 0,267 | 0,893 | 0,790 | 0,838 | 75 / 20 | 9 |

- Prag je F1-optimalan prag izabran na *out-of-fold* predviđanjima treninga.
  Testni skup na njega nema uticaja.
- Random Forest i XGBoost su **statistički izjednačeni**: razlika (0,006) je
  manja od četvrtine standardne devijacije. Sa samo 378 prevara za treniranje
  varijansa među podskupovima je velika (± 0,03), pa jedna podjela ne može
  razlučiti tako male razlike. Oba ansambla su jasno bolja od logističke
  regresije.
- Najbolji model je **≈ 490× bolji od slučajnog pogađanja** (0,841 / 0,0017).
- ROC-AUC (0,95–0,97) jedva razdvaja modele i čak ih rangira drugačije; PR-AUC
  ih razdvaja jasno. Zato se PR-AUC preferira pri ekstremnoj neuravnoteženosti.

**Najvažnija SHAP obilježja (Random Forest):** V12, V14, V4, V11, V3 — u skladu s
objavljenom literaturom o ovom skupu. Obilježja su PCA komponente, pa semantička
interpretacija nije moguća.

### 5.5 Tretman neuravnoteženosti klasa

`imbalance.py` poredi tri pristupa (5-struka CV na skupu za treniranje, SMOTE
uvijek unutar pipeline-a pa naduzorkuje samo podskupove za treniranje):

| Model | Bez korekcije | Težine klasa | SMOTE |
|---|---|---|---|
| Logistička regresija | **0,752** ± 0,012 | 0,719 ± 0,011 | 0,722 ± 0,007 |
| Random Forest | 0,836 ± 0,029 | **0,839** ± 0,023 | 0,834 ± 0,022 |
| XGBoost | 0,835 ± 0,036 | 0,853 ± 0,024 | **0,854** ± 0,027 |

(CV PR-AUC, ± std.) Nijedan pristup nije univerzalno bolji. Kod logističke
regresije korekcija čak šteti PR-AUC-u (pomjera vjerovatnoće, a ne rangiranje).
Kod XGBoost-a težine klasa i SMOTE donose ≈ +0,02, što je unutar jedne
standardne devijacije. SMOTE ne nadmašuje jednostavnije težine klasa, a skuplji
je.

### 5.6 Odabir praga osjetljiv na trošak

Maksimizacija F1 pretpostavlja da su obje greške jednako skupe. Ovdje se
propuštenoj prevari pripisuje **stvarni iznos transakcije**, a lažnoj uzbuni
**trošak ručnog pregleda od 3 €**. Prag se bira na out-of-fold predviđanjima
treninga, a trošak mjeri na testnom skupu (Random Forest):

| Strategija | Prag | Propušteno | Lažne uzbune | Ukupni trošak | Smanjenje |
|---|---|---|---|---|---|
| Bez modela | — | 95 | 0 | 9.920,44 € | — |
| F1-optimalan (trening) | 0,440 | 18 | 8 | 1.263,18 € | 87,3 % |
| **Troškovno optimalan (trening)** | **0,147** | **14** | **46** | **1.054,98 €** | **89,4 %** |
| *Optimum na testu (samo referenca)* | *0,369* | *16* | *9* | *945,98 €* | *90,5 %* |

Troškovno optimalan prag je ≈ 3× niži od F1-optimalnog: isplati se prihvatiti
38 lažnih uzbuna više (114 €) da bi se uhvatile 4 prevare više. Posljednji red
nije ostvariv u praksi (zahtijeva poznavanje testnog skupa) i prikazan je samo
da se vidi koliko je pravilno izabran prag blizu idealnog.

### 5.7 Nenadzirana trijaža prevara (hotelski podaci)

`hotel_bookings.csv` **nema oznaku prevare**, pa nadzirani model nije moguć niti
se može prijaviti ikakva tačnost. `fraud_detection.py` kombinuje šest domenskih
heuristika s algoritmom Isolation Forest u ocjenu rizika od 0 do 100
(60 % pravila / 40 % anomalija).

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

### 5.8 Pretraga hiperparametara

Prostori pretrage za sva četiri modela i najbolje vrijednosti nalaze se u
`reports/hotel_search_space.csv` i `reports/card_search_space.csv` (tabela
spremna za rad). Sažetak za hotelski zadatak:

| Model | Parametri (raspodjela) | Najbolje |
|---|---|---|
| Logistička regresija | C log-uniformno [0,001; 100], penalty {l1; l2}, class_weight | C = 1,15, l2, balanced |
| SVM (RBF) | C log-uniformno [0,1; 30], gamma log-uniformno [0,001; 0,3], class_weight | C = 25,8, gamma = 0,013 |
| Random Forest | n_estimators [100, 399], max_depth {10…25, None}, min_samples_leaf [1, 9], max_features {sqrt; log2; 0,3}, class_weight | 363 stabla, dubina 25, list 7, 0,3, balanced |
| XGBoost | n_estimators [200, 699], max_depth [3, 9], learning_rate log-uniformno [0,01; 0,3], subsample i colsample_bytree [0,6; 1], min_child_weight [1, 9], scale_pos_weight | 563 stabla, dubina 9, lr = 0,027 |

---

## 6. Ograničenja

- **Nasumična podjela je optimistična za otkazivanje.** Vremenska podjela daje
  ROC-AUC 0,78 umjesto 0,89 (§5.3). Glavni rezultati koriste nasumičnu podjelu
  radi poređenja modela, ali za stvarnu upotrebu je mjerodavniji vremenski broj.
- **SVM je treniran na uzorku od 20.000 redova** (RBF jezgro je kvadratne
  složenosti). Na punom skupu bi mogao biti nešto bolji, pa je njegov rezultat
  donja granica.
- **Uklanjanje duplikata** pretpostavlja da su identični redovi ista
  rezervacija upisana više puta. Neki mogu biti različite, ali identično opisane
  rezervacije (npr. grupe); tada je ukupni rezultat blago pesimističan.
- Pretraga hiperparametara je ograničena (15 kombinacija po modelu) zbog
  vremena izvršavanja.
- Trijaža hotelskih prevara je **nevalidirana** — bez stvarnih oznaka, njena
  preciznost i odziv su nepoznati.
- Skup podataka o karticama obuhvata dva dana transakcija iz 2013. godine iz
  jedne regije; rezultati se ne moraju prenijeti na druge periode ili sisteme.
- Analiza troška pretpostavlja fiksni trošak pregleda od 3 € i potpuni povrat
  spriječene prevare.
- Serija potražnje obuhvata ≈ 2 godine, što ograničava otkrivanje godišnje
  sezonalnosti.
- SHAP vrijednosti na podacima o karticama nisu semantički interpretabilne jer
  su V1–V28 PCA komponente.

## 7. Testovi

```bash
pip install pytest
python -m pytest tests/ -q
```

98 testova pokriva svojstva koja bi, ako se naruše, tiho poništila rezultate:

- **Zaštita od curenja podataka** — kolone `reservation_status` se uklanjaju;
  obilježja s dolaska su isključena; ciljna varijabla nikad nije obilježje;
  nijedno numeričko obilježje nema korelaciju > 0,95 s ciljem; i okidač koji
  traži svaku grupu (≥ 1 % redova) sa sigurnim ishodom — upravo tako je pronađeno
  curenje kroz parking.
- **Duplikati i priprema red po red** — duplikati se uklanjaju prije podjele, a
  rezultat pripreme jednog reda ne zavisi od ostalih redova (inače bi statistike
  testnog skupa procurile u treniranje).
- **Zaštita od gledanja unaprijed** — obilježja s pomakom u `demand_forecast.py`
  odgovaraju ispravno pomjerenoj seriji.
- **Protokol modeliranja** — svaki od 4 modela se trenira na oba zadatka; svaki
  ključ prostora pretrage postoji u pipeline-u; out-of-fold predviđanja pokrivaju
  sve redove; rijetke i neviđene kategorije idu u istu kolonu.
- **Analiza troška** — brzi proračun troška poklapa se s naivnom referentnom
  implementacijom, i izabrani prag je zaista najjeftiniji.
- **Regresijski test tipova podataka** — pandas ≥ 2 tekst čuva kao `string`, pa
  naivna provjera `dtype == object` pogrešno klasifikuje kategorijske kolone.

## 8. Reproducibilnost

Sve nasumične operacije koriste fiksno sjeme (`RANDOM_STATE = 42`). Podjele su
stratifikovane (klasifikacija) ili hronološke (vremenske serije). Rezultati su
dobijeni s Python 3.12 i: scikit-learn 1.9.0, xgboost 3.4.1, shap 0.52.0,
imbalanced-learn 0.14.2, pandas 3.0.5, numpy 2.5.2, scipy 1.18.0,
statsmodels 0.14.6, matplotlib 3.11.1, seaborn 0.13.2.

## 9. Napomena o izradi

Programsko rješenje razvijeno je na osnovu javno dostupnih primjera i
dokumentacije (vidjeti literaturu u radu, [22]–[29]), uz pomoć AI asistenata
ChatGPT (OpenAI) i Claude (Anthropic) pri pisanju i doradi koda. Izbor podataka i
metodologije, pokretanje analize, provjera rezultata i zaključci rezultat su rada
autorice.

## 10. Literatura

- Antonio, N., de Almeida, A., & Nunes, L. (2019). *Hotel booking demand datasets.*
  Data in Brief, 22, 41–49.
- Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2015).
  *Calibrating probability with undersampling for unbalanced classification.* IEEE SSCI.
- Lundberg, S. M., & Lee, S.-I. (2017). *A unified approach to interpreting model
  predictions.* NeurIPS.
- Chawla, N. V. et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique.* JAIR.
