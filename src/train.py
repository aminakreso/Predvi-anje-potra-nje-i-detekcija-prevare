"""Train and evaluate cancellation-prediction models on the hotel bookings data.

Usage:
    python src/train.py --data-path data/raw/hotel_bookings.csv
"""

import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_prep import TARGET, get_feature_lists, load_and_prepare
from plotting import FIG_DIR, add_show_argument, emit, plt, report_saved

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = ROOT / "data" / "raw" / "hotel_bookings.csv"
MODEL_DIR = ROOT / "models"
RANDOM_STATE = 42

warnings.filterwarnings("ignore", category=UserWarning)


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )


def build_models() -> dict:
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    except ImportError:
        models["gradient_boosting"] = HistGradientBoostingClassifier(
            random_state=RANDOM_STATE
        )

    return models


def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def plot_roc_curves(fitted_pipelines: dict, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, pipeline in fitted_pipelines.items():
        RocCurveDisplay.from_estimator(pipeline, X_test, y_test, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_title("ROC curves (test set)")
    ax.legend(loc="lower right")
    emit(fig, "07_roc_curves.png")


def plot_feature_importance(pipeline: Pipeline, model_name: str, top_n: int = 20) -> None:
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = pd.Series(model.feature_importances_, index=feature_names)
    top = importances.sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(8, 8))
    top.sort_values().plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_title(f"Top {top_n} feature importances ({model_name})")
    emit(fig, "08_feature_importance.png")


def plot_confusion_matrix(y_test: pd.Series, y_pred: np.ndarray, model_name: str) -> None:
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xticks([0, 1], ["Not canceled", "Canceled"])
    ax.set_yticks([0, 1], ["Not canceled", "Canceled"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix ({model_name})")
    fig.colorbar(im, ax=ax)
    emit(fig, "09_confusion_matrix.png")


def main(data_path: Path) -> None:
    print(f"Loading data from {data_path} ...")
    df = load_and_prepare(data_path)
    numeric_cols, categorical_cols = get_feature_lists(df)

    X = df[numeric_cols + categorical_cols]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train rows: {len(X_train)}  Test rows: {len(X_test)}  "
          f"Cancellation rate (train): {y_train.mean():.3f}")

    models = build_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = {}
    fitted_pipelines = {}

    for name, model in models.items():
        # A fresh preprocessor per model -- sharing one instance across
        # pipelines would leave every saved model pointing at the same
        # fitted object.
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor(numeric_cols, categorical_cols)),
            ("model", model),
        ])

        cv_scores = cross_val_score(
            pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
        )
        print(f"[{name}] 5-fold CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline
        results[name] = evaluate(pipeline, X_test, y_test)
        results[name]["cv_roc_auc_mean"] = cv_scores.mean()

    results_df = pd.DataFrame(results).T.sort_values("roc_auc", ascending=False)
    print("\nTest-set results:")
    print(results_df.round(4).to_string())

    best_name = results_df.index[0]
    best_pipeline = fitted_pipelines[best_name]
    print(f"\nBest model: {best_name}")
    print(classification_report(y_test, best_pipeline.predict(X_test),
                                 target_names=["Not canceled", "Canceled"]))

    plot_roc_curves(fitted_pipelines, X_test, y_test)
    plot_feature_importance(best_pipeline, best_name)
    plot_confusion_matrix(y_test, best_pipeline.predict(X_test), best_name)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_DIR / "best_model.joblib", compress=3)
    results_df.to_csv(ROOT / "reports" / "model_comparison.csv")
    print(f"\nSaved best model to {MODEL_DIR / 'best_model.joblib'}")
    print(report_saved())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    add_show_argument(parser)
    args = parser.parse_args()
    main(args.data_path)
