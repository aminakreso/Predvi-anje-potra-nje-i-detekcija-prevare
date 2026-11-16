"""Hyperparameter search for all four models on both tasks.

RandomizedSearchCV over the spaces in modeling.search_space(), run on the
TRAINING split only (the test split is never seen during tuning). Writes:

  reports/{task}_best_params.json  -- read by train.py / card_fraud.py
  reports/{task}_search_space.csv  -- search space, chosen values and CV score
                                      of the best combination (thesis Table 4)

Run this before train.py and card_fraud.py. Re-running with --models only
updates those models' entries, so one model can be re-tuned on its own.

Usage:
    python src/tuning.py --task both
    python src/tuning.py --task hotel --models svm xgboost --n-iter 25
"""

import argparse
import time
import warnings
from pathlib import Path

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from modeling import (
    MODEL_LABELS,
    MODEL_NAMES,
    RANDOM_STATE,
    build_pipeline,
    describe_distribution,
    format_value,
    load_params,
    save_params,
    search_space,
    split_data,
)

ROOT = Path(__file__).resolve().parents[1]
HOTEL_DATA = ROOT / "data" / "raw" / "hotel_bookings.csv"
CARD_DATA = ROOT / "data" / "raw" / "creditcard.csv"
REPORT_DIR = ROOT / "reports"
# Metrika po kojoj se bira najbolja kombinacija parametara – ista kao metrika
# po kojoj se poslije bira najbolji model:
#   hotel -> ROC-AUC (28 % otkazivanja, klase su dovoljno uravnotežene)
#   card  -> PR-AUC ("average_precision"), jer je prevara samo 0,17 %,
#            pa ROC-AUC izgleda odlično i kod slabih modela.
SCORING = {"hotel": "roc_auc", "card": "average_precision"}

warnings.filterwarnings("ignore")


# Učitava podatke i vraća SAMO dio za treniranje (80 %).
# Zašto: testni skup (20 %) se ne smije koristiti pri izboru parametara, inače
# bi parametri bili "naštimani" baš na test, a rezultat na testu optimističan.
# split_data() koristi isti random_state kao train.py i card_fraud.py, pa je
# ovo potpuno isti trening skup na kojem se modeli kasnije i treniraju.
def load_training_split(task: str):
    if task == "hotel":
        from data_prep import load_modeling_data

        X, y, numeric_cols, categorical_cols = load_modeling_data(HOTEL_DATA)
    else:
        from card_fraud import load_modeling_data

        X, y = load_modeling_data(CARD_DATA)
        numeric_cols, categorical_cols = list(X.columns), []
    X_train, _, y_train, _ = split_data(X, y)
    return X_train, y_train, numeric_cols, categorical_cols


# Glavna funkcija: za svaki model traži najbolje hiperparametre za jedan zadatak.
# Postupak (slučajna pretraga, RandomizedSearchCV):
#   1. iz prostora pretrage (modeling.search_space) nasumično izvuče n_iter
#      kombinacija parametara (podrazumijevano 15);
#   2. svaku kombinaciju ocijeni 3-strukom stratifikovanom unakrsnom validacijom
#      na trening skupu;
#   3. zapamti kombinaciju s najboljom prosječnom CV ocjenom.
# Rezultat se upisuje u reports/{task}_best_params.json (čitaju ga train.py i
# card_fraud.py) i reports/{task}_search_space.csv (tabela za rad).
def tune_task(task: str, models: list[str], n_iter: int, cv_folds: int) -> None:
    print("\n" + "=" * 70)
    print(f"{task.upper()} -- tuning (scoring: {SCORING[task]})")
    print("=" * 70)

    X_train, y_train, numeric_cols, categorical_cols = load_training_split(task)
    params_path = REPORT_DIR / f"{task}_best_params.json"
    table_path = REPORT_DIR / f"{task}_search_space.csv"
    # Učitavaju se postojeći rezultati, da pokretanje s --models svm prepiše
    # samo SVM, a ostale modele ostavi kakvi jesu.
    best_params = load_params(params_path)
    table = pd.read_csv(table_path) if table_path.exists() else pd.DataFrame()

    for name in models:
        # Pipeline = priprema podataka (skaliranje, one-hot) + model. Priprema je
        # dio pipeline-a, pa se u svakom CV preklopu uči samo na tom preklopu
        # (nema curenja podataka iz validacijskog dijela).
        # The search runs candidates in parallel, so each model uses one core.
        pipeline = build_pipeline(name, task, numeric_cols, categorical_cols, y_train, n_jobs=1)
        # Rasponi parametara, npr. learning_rate log-uniformno iz [0,01; 0,3].
        space = search_space(name, task, y_train)
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=space,
            n_iter=n_iter,
            scoring=SCORING[task],
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE),
            random_state=RANDOM_STATE,
            n_jobs=-1,
            refit=False,  # train.py / card_fraud.py refit with the chosen values
        )
        start = time.time()
        # Ovdje se izvršava pretraga: n_iter kombinacija × cv_folds treniranja.
        search.fit(X_train, y_train)
        # Najbolja kombinacija i njena CV ocjena (prosjek ± std preko preklopa).
        best = search.best_index_
        mean = search.cv_results_["mean_test_score"][best]
        std = search.cv_results_["std_test_score"][best]
        print(f"\n[{name}] best CV {SCORING[task]}: {mean:.4f} +/- {std:.4f} "
              f"({n_iter} candidates x {cv_folds} folds, {time.time() - start:.0f}s)")

        # Imena parametara u pipeline-u imaju prefiks "model__"; on se skida
        # da bi ih train.py mogao direktno predati modelu.
        chosen = {key.removeprefix("model__"): value
                  for key, value in search.best_params_.items()}
        for key, value in chosen.items():
            print(f"  {key}: {value}")
        best_params[name] = chosen

        # Jedan red po parametru: raspon pretrage, izabrana vrijednost i CV
        # ocjena – to je tabela prostora pretrage u radu.
        rows = [{
            "model": MODEL_LABELS[name],
            "parameter": key.removeprefix("model__").removeprefix("estimator__"),
            "search_space": describe_distribution(dist),
            "best_value": format_value(search.best_params_[key]),
            "best_cv_score_mean": mean,
            "best_cv_score_std": std,
            "metric": SCORING[task],
        } for key, dist in space.items()]
        if not table.empty:
            table = table[table["model"] != MODEL_LABELS[name]]
        table = pd.concat([table, pd.DataFrame(rows)], ignore_index=True)

        # Saved after every model, so a long run that is interrupted keeps
        # everything finished so far.
        save_params(params_path, best_params)
        order = {MODEL_LABELS[m]: i for i, m in enumerate(MODEL_NAMES)}
        table.sort_values("model", key=lambda s: s.map(order), kind="stable").to_csv(
            table_path, index=False)

    print(f"\nSaved {params_path.name} and {table_path.name} to {REPORT_DIR}")


# Pokreće pretragu za izabrani zadatak (hotel, card ili oba redom).
def main(task: str, models: list[str], n_iter: int, cv_folds: int) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for t in (["hotel", "card"] if task == "both" else [task]):
        tune_task(t, models, n_iter, cv_folds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["hotel", "card", "both"], default="both")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=MODEL_NAMES)
    parser.add_argument("--n-iter", type=int, default=15,
                        help="RandomizedSearchCV candidates per model (higher = slower, better)")
    parser.add_argument("--cv-folds", type=int, default=3)
    args = parser.parse_args()
    main(args.task, args.models, args.n_iter, args.cv_folds)
