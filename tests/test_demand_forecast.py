"""Tests for demand_forecast -- the lookahead tests matter most, since a lag
feature that peeks at the present invalidates every forecast metric."""

import numpy as np
import pandas as pd
import pytest

from demand_forecast import (
    SEASONAL_PERIOD,
    add_time_features,
    build_daily_series,
    forecast_naive,
    forecast_seasonal_naive,
    score,
)


@pytest.fixture
def daily_series() -> pd.Series:
    index = pd.date_range("2016-01-01", periods=120, freq="D")
    values = 100 + 10 * np.sin(np.arange(120) * 2 * np.pi / 7) + np.arange(120) * 0.1
    return pd.Series(values, index=index, name="y")


class TestDailySeries:
    def test_calendar_has_no_gaps(self, prepared_hotel_df):
        daily = build_daily_series(prepared_hotel_df)
        expected = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
        assert daily.index.equals(expected), "missing days would corrupt lag features"

    def test_net_equals_gross_minus_cancellations(self, prepared_hotel_df):
        daily = build_daily_series(prepared_hotel_df)
        expected = daily["gross_bookings"] - daily["cancellations"]
        pd.testing.assert_series_equal(
            daily["net_bookings"], expected, check_names=False
        )

    def test_counts_are_non_negative(self, prepared_hotel_df):
        daily = build_daily_series(prepared_hotel_df)
        assert (daily[["gross_bookings", "cancellations", "net_bookings"]] >= 0).all().all()

    def test_gross_total_matches_row_count(self, prepared_hotel_df):
        daily = build_daily_series(prepared_hotel_df)
        assert daily["gross_bookings"].sum() == len(prepared_hotel_df)

    def test_unknown_hotel_raises(self, prepared_hotel_df):
        with pytest.raises(SystemExit):
            build_daily_series(prepared_hotel_df, hotel="Nonexistent Hotel")


class TestLagFeatures:
    def test_lag_columns_match_shifted_values(self, daily_series):
        frame = add_time_features(daily_series)
        for lag in (1, 2, 7, 14, 28):
            expected = daily_series.shift(lag).loc[frame.index]
            pd.testing.assert_series_equal(
                frame[f"lag_{lag}"], expected, check_names=False
            )

    def test_no_lag_uses_current_value(self, daily_series):
        """The defining anti-lookahead property: no feature may equal y[t]."""
        frame = add_time_features(daily_series)
        feature_cols = [c for c in frame.columns if c != "y"]
        for column in feature_cols:
            if frame[column].dtype.kind not in "fi":
                continue
            assert not np.allclose(frame[column], frame["y"]), (
                f"{column} reproduces the current target -- lookahead leak"
            )

    def test_rolling_stats_exclude_current_day(self, daily_series):
        frame = add_time_features(daily_series)
        expected = daily_series.shift(1).rolling(7).mean().loc[frame.index]
        pd.testing.assert_series_equal(
            frame["roll_mean_7"], expected, check_names=False
        )

    def test_rows_with_incomplete_lags_are_dropped(self, daily_series):
        frame = add_time_features(daily_series)
        assert not frame.isna().any().any()
        # 28-day lag + 28-day rolling window is the binding constraint.
        assert len(frame) == len(daily_series) - 28

    def test_calendar_features_correct(self, daily_series):
        frame = add_time_features(daily_series)
        assert (frame["dayofweek"] == frame.index.dayofweek).all()
        assert (frame["month"] == frame.index.month).all()
        assert (frame["is_weekend"] == (frame.index.dayofweek >= 5).astype(int)).all()


class TestBaselineForecasts:
    def test_naive_forecast_is_previous_value(self, daily_series):
        train, test = daily_series.iloc[:-10], daily_series.iloc[-10:]
        predictions = forecast_naive(train, test)
        expected = daily_series.shift(1).loc[test.index].to_numpy()
        np.testing.assert_allclose(predictions, expected)

    def test_naive_first_prediction_uses_last_training_point(self, daily_series):
        train, test = daily_series.iloc[:-10], daily_series.iloc[-10:]
        assert forecast_naive(train, test)[0] == pytest.approx(train.iloc[-1])

    def test_seasonal_naive_uses_same_weekday(self, daily_series):
        train, test = daily_series.iloc[:-10], daily_series.iloc[-10:]
        predictions = forecast_seasonal_naive(train, test)
        expected = daily_series.shift(SEASONAL_PERIOD).loc[test.index].to_numpy()
        np.testing.assert_allclose(predictions, expected)

    def test_forecast_length_matches_test(self, daily_series):
        train, test = daily_series.iloc[:-10], daily_series.iloc[-10:]
        assert len(forecast_naive(train, test)) == len(test)
        assert len(forecast_seasonal_naive(train, test)) == len(test)


class TestScoring:
    def test_perfect_forecast_scores_zero(self):
        y = np.array([10.0, 20.0, 30.0])
        result = score(y, y)
        assert result["MAE"] == pytest.approx(0.0)
        assert result["RMSE"] == pytest.approx(0.0)
        assert result["MAPE_%"] == pytest.approx(0.0)

    def test_mae_matches_manual_calculation(self):
        y_true = np.array([10.0, 20.0])
        y_pred = np.array([12.0, 18.0])
        assert score(y_true, y_pred)["MAE"] == pytest.approx(2.0)

    def test_rmse_penalises_large_errors_more_than_mae(self):
        y_true = np.array([0.0, 0.0, 0.0, 0.0])
        y_pred = np.array([0.0, 0.0, 0.0, 8.0])
        result = score(y_true, y_pred)
        assert result["RMSE"] > result["MAE"]
