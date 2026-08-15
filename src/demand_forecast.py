"""Demand forecasting for hotel bookings (time-series half of the project).

Where train.py predicts cancellation risk for an INDIVIDUAL booking, this
module forecasts AGGREGATE demand over time: how many guests actually arrive
per day. The two halves connect directly --

    net demand = gross bookings - cancellations

-- so the cancellation model from train.py is what turns a gross booking
count into the net occupancy figure a hotel actually plans capacity against.

Evaluation protocol: walk-forward ONE-STEP-AHEAD backtest on a time-based
split (never a random split -- that would let the model peek at the future).
Every model is scored under the identical protocol so the numbers compare.

Usage:
    python src/demand_forecast.py
    python src/demand_forecast.py --hotel "City Hotel" --test-days 120
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX

from data_prep import TARGET, load_and_prepare
from plotting import FIG_DIR, add_show_argument, emit, plt, report_saved

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = ROOT / "data" / "raw" / "hotel_bookings.csv"
REPORT_DIR = ROOT / "reports"
RANDOM_STATE = 42
SEASONAL_PERIOD = 7  # weekly seasonality dominates daily hotel arrivals

warnings.filterwarnings("ignore")


def build_daily_series(df: pd.DataFrame, hotel: str | None = None) -> pd.DataFrame:
    """Collapse individual bookings into a daily arrivals time series."""
    if hotel:
        df = df[df["hotel"] == hotel]
        if df.empty:
            raise SystemExit(f"No rows for hotel={hotel!r}")

    dates = pd.to_datetime(
        dict(
            year=df["arrival_date_year"],
            month=df["arrival_month_num"],
            day=df["arrival_date_day_of_month"],
        ),
        errors="coerce",
    )
    work = pd.DataFrame({
        "date": dates,
        "canceled": df[TARGET].to_numpy(),
        "guests": df["total_guests"].to_numpy(),
    }).dropna(subset=["date"])

    daily = work.groupby("date").agg(
        gross_bookings=("canceled", "size"),
        cancellations=("canceled", "sum"),
        gross_guests=("guests", "sum"),
    )
    daily["net_bookings"] = daily["gross_bookings"] - daily["cancellations"]
    daily["cancellation_rate"] = daily["cancellations"] / daily["gross_bookings"]

    # Reindex onto a complete calendar so gaps become explicit zeros rather
    # than silently-missing days (which would corrupt lag features).
    full_index = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_index).fillna(0)
    daily.index.name = "date"

    return daily


def add_time_features(series: pd.Series) -> pd.DataFrame:
    """Lag + calendar features. All lags are >= 1 day, so nothing leaks."""
    frame = pd.DataFrame({"y": series})

    for lag in (1, 2, 7, 14, 28):
        frame[f"lag_{lag}"] = series.shift(lag)
    # Rolling stats are shifted by 1 so the current day is never included.
    for window in (7, 28):
        frame[f"roll_mean_{window}"] = series.shift(1).rolling(window).mean()
        frame[f"roll_std_{window}"] = series.shift(1).rolling(window).std()

    frame["dayofweek"] = series.index.dayofweek
    frame["month"] = series.index.month
    frame["weekofyear"] = series.index.isocalendar().week.astype(int)
    frame["is_weekend"] = (series.index.dayofweek >= 5).astype(int)

    return frame.dropna()


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE_%": mean_absolute_percentage_error(y_true, y_pred) * 100,
    }


def forecast_naive(train: pd.Series, test: pd.Series) -> np.ndarray:
    """Baseline: tomorrow looks like today."""
    history = pd.concat([train, test])
    return history.shift(1).loc[test.index].to_numpy()


def forecast_seasonal_naive(train: pd.Series, test: pd.Series) -> np.ndarray:
    """Baseline: this Tuesday looks like last Tuesday. Hard to beat on daily data."""
    history = pd.concat([train, test])
    return history.shift(SEASONAL_PERIOD).loc[test.index].to_numpy()


def forecast_sarima(train: pd.Series, test: pd.Series) -> np.ndarray:
    """SARIMA, refit once then updated observation-by-observation.

    `append(refit=False)` feeds each true observation back in so the next
    prediction is genuinely one-step-ahead, without paying to refit daily.
    """
    model = SARIMAX(
        train,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, SEASONAL_PERIOD),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)

    predictions = []
    for timestamp in test.index:
        predictions.append(float(res.forecast(steps=1).iloc[0]))
        res = res.append(test.loc[[timestamp]], refit=False)

    return np.array(predictions)


def forecast_ml(series: pd.Series, test_index: pd.DatetimeIndex) -> np.ndarray:
    """Gradient boosting on lag/calendar features, one-step-ahead.

    The model is fit only on data strictly before the test window; at
    prediction time it consumes TRUE past values (standard one-step-ahead
    backtest), matching how the other models here are scored.
    """
    frame = add_time_features(series)
    feature_cols = [c for c in frame.columns if c != "y"]

    train_frame = frame[frame.index < test_index[0]]
    test_frame = frame[frame.index.isin(test_index)]

    model = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.05, random_state=RANDOM_STATE
    )
    model.fit(train_frame[feature_cols], train_frame["y"])

    return model.predict(test_frame[feature_cols])


def plot_series(daily: pd.DataFrame, label: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(daily.index, daily["gross_bookings"], label="Gross bookings", alpha=0.6)
    ax.plot(daily.index, daily["net_bookings"], label="Net (after cancellations)")
    ax.set_ylabel("Bookings per day")
    ax.set_title(f"Daily hotel demand -- {label}")
    ax.legend()
    emit(fig, "14_demand_daily_series.png")


def plot_decomposition(series: pd.Series) -> None:
    result = seasonal_decompose(series, model="additive", period=SEASONAL_PERIOD)
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    for ax, (data, name) in zip(axes, [
        (result.observed, "Observed"),
        (result.trend, "Trend"),
        (result.seasonal, "Seasonal (weekly)"),
        (result.resid, "Residual"),
    ]):
        ax.plot(data.index, data)
        ax.set_ylabel(name, fontsize=9)
    axes[0].set_title("Seasonal decomposition of net daily demand")
    emit(fig, "15_demand_decomposition.png")


def plot_forecasts(train: pd.Series, test: pd.Series, forecasts: dict) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    context = train.iloc[-60:]
    ax.plot(context.index, context, color="gray", label="History", alpha=0.6)
    ax.plot(test.index, test, color="black", linewidth=2, label="Actual")
    for name, values in forecasts.items():
        ax.plot(test.index, values, alpha=0.8, label=name)
    ax.set_ylabel("Net bookings per day")
    ax.set_title("One-step-ahead forecasts vs. actual (test window)")
    ax.legend(fontsize=8)
    emit(fig, "16_demand_forecast_vs_actual.png")


def plot_monthly_seasonality(daily: pd.DataFrame) -> None:
    by_month = daily.groupby(daily.index.month)["net_bookings"].mean()
    by_dow = daily.groupby(daily.index.dayofweek)["net_bookings"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    by_month.plot(kind="bar", ax=axes[0], color="#4C72B0")
    axes[0].set_title("Average net demand by month")
    axes[0].set_xlabel("Month")
    by_dow.plot(kind="bar", ax=axes[1], color="#55A868")
    axes[1].set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], rotation=0)
    axes[1].set_title("Average net demand by day of week")
    axes[1].set_xlabel("")
    emit(fig, "17_demand_seasonality.png")


def main(data_path: Path, hotel: str | None, test_days: int) -> None:
    print(f"Loading data from {data_path} ...")
    df = load_and_prepare(data_path)
    label = hotel or "both hotels"
    daily = build_daily_series(df, hotel)

    print(f"Daily series: {daily.index.min().date()} to {daily.index.max().date()} "
          f"({len(daily)} days, {label})")
    print(f"Mean gross bookings/day: {daily['gross_bookings'].mean():.1f}")
    print(f"Mean net bookings/day:   {daily['net_bookings'].mean():.1f}")
    print(f"Overall cancellation rate: {daily['cancellations'].sum() / daily['gross_bookings'].sum():.3f}")

    series = daily["net_bookings"].astype(float)

    # Time-based split -- the final `test_days` are held out chronologically.
    split_point = len(series) - test_days
    train, test = series.iloc[:split_point], series.iloc[split_point:]
    print(f"\nTrain: {train.index.min().date()} to {train.index.max().date()} ({len(train)} days)")
    print(f"Test:  {test.index.min().date()} to {test.index.max().date()} ({len(test)} days)")

    print("\nRunning one-step-ahead backtests ...")
    forecasts = {
        "naive": forecast_naive(train, test),
        "seasonal_naive": forecast_seasonal_naive(train, test),
        "sarima": forecast_sarima(train, test),
        "gradient_boosting": forecast_ml(series, test.index),
    }

    results = {name: score(test.to_numpy(), pred) for name, pred in forecasts.items()}
    results_df = pd.DataFrame(results).T.sort_values("MAE")

    print("\nForecast accuracy (lower is better, ranked by MAE):")
    print(results_df.round(3).to_string())

    best = results_df.index[0]
    naive_mae = results_df.loc["seasonal_naive", "MAE"]
    best_mae = results_df.loc[best, "MAE"]
    improvement = (naive_mae - best_mae) / naive_mae * 100
    print(f"\nBest model: {best}")
    print(f"  Improvement over seasonal-naive baseline: {improvement:+.1f}% MAE")
    if improvement <= 0:
        print("  NOTE: nothing beat the naive baseline -- report this honestly, "
              "it is a real and common result on short daily series.")

    plot_series(daily, label)
    plot_decomposition(series)
    plot_forecasts(train, test, forecasts)
    plot_monthly_seasonality(daily)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(REPORT_DIR / "demand_forecast_comparison.csv")
    daily.to_csv(REPORT_DIR / "daily_demand_series.csv")
    print(f"\nSaved results to {REPORT_DIR / 'demand_forecast_comparison.csv'}")
    print(report_saved())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--hotel", type=str, default=None,
                        help='Filter to one hotel, e.g. "City Hotel" or "Resort Hotel"')
    parser.add_argument("--test-days", type=int, default=90)
    add_show_argument(parser)
    args = parser.parse_args()
    main(args.data_path, args.hotel, args.test_days)
