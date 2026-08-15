"""Tests for the two fraud modules: supervised (card_fraud) and unsupervised
(fraud_detection), plus the cost-curve logic in explain."""

import numpy as np
import pandas as pd
import pytest

from card_fraud import best_f1_threshold, engineer_features
from explain import cost_curve
from fraud_detection import add_rule_flags, combine_scores


@pytest.fixture
def raw_card_df() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = 200
    df = pd.DataFrame(rng.normal(size=(n, 28)), columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", rng.integers(0, 172800, n))
    df["Amount"] = rng.uniform(0, 500, n).round(2)
    df["Class"] = (rng.random(n) < 0.05).astype(int)
    return df


class TestCardFeatureEngineering:
    def test_raw_time_and_amount_replaced(self, raw_card_df):
        result = engineer_features(raw_card_df)
        assert "Time" not in result.columns
        assert "Amount" not in result.columns
        assert "hour" in result.columns
        assert "log_amount" in result.columns

    def test_hour_within_valid_range(self, raw_card_df):
        result = engineer_features(raw_card_df)
        assert result["hour"].between(0, 23).all()

    def test_log_amount_is_invertible(self, raw_card_df):
        """explain.py recovers EUR amounts via expm1, so the round-trip must hold."""
        result = engineer_features(raw_card_df)
        recovered = np.expm1(result["log_amount"].to_numpy())
        np.testing.assert_allclose(recovered, raw_card_df["Amount"].to_numpy(), rtol=1e-9)

    def test_target_preserved(self, raw_card_df):
        result = engineer_features(raw_card_df)
        pd.testing.assert_series_equal(result["Class"], raw_card_df["Class"])

    def test_pca_components_untouched(self, raw_card_df):
        result = engineer_features(raw_card_df)
        for column in [f"V{i}" for i in range(1, 29)]:
            pd.testing.assert_series_equal(result[column], raw_card_df[column])


class TestThresholdSelection:
    def test_perfect_separation_gives_f1_of_one(self):
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.8, 0.9])
        threshold, f1 = best_f1_threshold(y_true, y_proba)
        assert f1 == pytest.approx(1.0)
        assert 0.2 < threshold <= 0.8

    def test_threshold_beats_default_when_scores_are_shifted(self):
        """All probabilities sit below 0.5, so the default cutoff would predict
        nothing; the tuned threshold must still recover the positives."""
        y_true = np.array([0, 0, 0, 1, 1])
        y_proba = np.array([0.01, 0.02, 0.03, 0.20, 0.25])
        threshold, f1 = best_f1_threshold(y_true, y_proba)
        assert threshold < 0.5
        assert f1 == pytest.approx(1.0)

    def test_returned_threshold_reproduces_reported_f1(self):
        rng = np.random.default_rng(3)
        y_true = (rng.random(300) < 0.2).astype(int)
        y_proba = np.clip(y_true * 0.4 + rng.normal(0.3, 0.2, 300), 0, 1)
        threshold, f1 = best_f1_threshold(y_true, y_proba)

        predicted = (y_proba >= threshold).astype(int)
        tp = int(((predicted == 1) & (y_true == 1)).sum())
        fp = int(((predicted == 1) & (y_true == 0)).sum())
        fn = int(((predicted == 0) & (y_true == 1)).sum())
        recomputed = 2 * tp / (2 * tp + fp + fn)
        assert recomputed == pytest.approx(f1, abs=1e-6)


class TestCostCurve:
    def test_missed_fraud_costs_transaction_amount(self):
        y_true = np.array([1, 1])
        y_proba = np.array([0.01, 0.01])
        amounts = np.array([100.0, 250.0])
        costs = cost_curve(y_true, y_proba, amounts, review_cost=3.0)

        # Above the scores both frauds go undetected: cost = 100 + 250.
        missed_all = costs[costs["threshold"] > 0.01]
        assert missed_all["total_cost"].unique() == pytest.approx([350.0])
        # Below them both are caught, so nothing is lost.
        caught_all = costs[costs["threshold"] <= 0.01]
        assert caught_all["total_cost"].unique() == pytest.approx([0.0])
        # There are no legitimate rows, so a false alarm is impossible.
        assert costs["review_cost"].max() == pytest.approx(0.0)

    def test_false_alarms_cost_review_fee(self):
        y_true = np.array([0, 0, 0, 0])
        y_proba = np.array([0.99, 0.99, 0.99, 0.99])
        amounts = np.array([10.0, 10.0, 10.0, 10.0])
        costs = cost_curve(y_true, y_proba, amounts, review_cost=2.5)
        assert costs["total_cost"].max() == pytest.approx(10.0)  # 4 alarms x 2.50
        assert costs["fraud_loss"].max() == pytest.approx(0.0)

    def test_total_is_sum_of_components(self):
        rng = np.random.default_rng(4)
        y_true = (rng.random(100) < 0.1).astype(int)
        y_proba = rng.random(100)
        amounts = rng.uniform(10, 500, 100)
        costs = cost_curve(y_true, y_proba, amounts, review_cost=3.0)
        np.testing.assert_allclose(
            costs["total_cost"], costs["fraud_loss"] + costs["review_cost"]
        )

    def test_raising_threshold_never_reduces_missed_frauds(self):
        rng = np.random.default_rng(5)
        y_true = (rng.random(200) < 0.15).astype(int)
        y_proba = rng.random(200)
        amounts = rng.uniform(10, 500, 200)
        costs = cost_curve(y_true, y_proba, amounts, review_cost=3.0)
        assert costs["frauds_missed"].is_monotonic_increasing
        assert costs["false_alarms"].is_monotonic_decreasing


class TestFraudRules:
    def test_flags_are_binary(self, prepared_hotel_df):
        flagged = add_rule_flags(prepared_hotel_df)
        for column in [c for c in flagged.columns if c.startswith("flag_")]:
            assert set(flagged[column].unique()) <= {0, 1}

    def test_rule_score_equals_flag_sum(self, prepared_hotel_df):
        flagged = add_rule_flags(prepared_hotel_df)
        flag_cols = [c for c in flagged.columns if c.startswith("flag_")]
        pd.testing.assert_series_equal(
            flagged["rule_score"], flagged[flag_cols].sum(axis=1), check_names=False
        )

    def test_rules_do_not_drop_or_reorder_rows(self, prepared_hotel_df):
        flagged = add_rule_flags(prepared_hotel_df)
        assert len(flagged) == len(prepared_hotel_df)
        assert flagged.index.equals(prepared_hotel_df.index)

    def test_serial_canceller_rule_matches_definition(self, prepared_hotel_df):
        flagged = add_rule_flags(prepared_hotel_df)
        expected = (
            (flagged["previous_cancellations"] >= 3)
            & (flagged["previous_bookings_not_canceled"] == 0)
        ).astype(int)
        pd.testing.assert_series_equal(
            flagged["flag_serial_canceller"], expected, check_names=False
        )

    def test_risk_score_stays_within_zero_to_hundred(self, prepared_hotel_df):
        flagged = add_rule_flags(prepared_hotel_df)
        flagged["anomaly_score"] = np.linspace(-0.2, 0.3, len(flagged))
        scored = combine_scores(flagged)
        assert scored["risk_score"].between(0, 100).all()
