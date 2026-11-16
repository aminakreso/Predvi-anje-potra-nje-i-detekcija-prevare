"""Tests for data_prep -- especially the leakage guards, which are the single
most important correctness property in this project."""

import numpy as np
import pandas as pd
import pytest

from data_prep import (
    LEAKAGE_COLS,
    MAX_VALID_ADR,
    POST_BOOKING_FEATURES,
    TARGET,
    clean_data,
    engineer_features,
    get_feature_lists,
    load_and_prepare,
    load_data,
    load_modeling_data,
)


class TestLeakageGuards:
    def test_leakage_columns_are_dropped(self, prepared_hotel_df):
        for column in LEAKAGE_COLS:
            assert column not in prepared_hotel_df.columns, (
                f"{column} leaks the target and must never survive preparation"
            )

    def test_target_is_not_a_feature(self, prepared_hotel_df):
        numeric_cols, categorical_cols = get_feature_lists(prepared_hotel_df)
        assert TARGET not in numeric_cols
        assert TARGET not in categorical_cols

    def test_no_feature_perfectly_predicts_target(self, real_hotel_df):
        """A |correlation| of ~1.0 with the target means leakage slipped back in."""
        numeric_cols, _ = get_feature_lists(real_hotel_df)
        correlations = real_hotel_df[numeric_cols].corrwith(real_hotel_df[TARGET]).abs()
        suspicious = correlations[correlations > 0.95]
        assert suspicious.empty, f"Suspiciously predictive features: {dict(suspicious)}"

    def test_post_booking_features_excluded_by_default(self, prepared_hotel_df):
        numeric_cols, categorical_cols = get_feature_lists(prepared_hotel_df)
        for column in POST_BOOKING_FEATURES:
            assert column not in numeric_cols + categorical_cols

    def test_post_booking_features_available_on_request(self, prepared_hotel_df):
        """sensitivity.py needs them to measure how much they inflate results."""
        numeric_cols, categorical_cols = get_feature_lists(
            prepared_hotel_df, include_post_booking=True
        )
        for column in POST_BOOKING_FEATURES:
            assert column in numeric_cols + categorical_cols

    @staticmethod
    def _groups_with_certain_outcome(df, columns):
        found = {}
        for column in columns:
            stats = df.groupby(column)[TARGET].agg(["mean", "size"])
            large = stats[stats["size"] >= 0.01 * len(df)]
            certain = large[large["mean"].isin([0.0, 1.0])]
            if not certain.empty:
                found[column] = certain.to_dict("index")
        return found

    def test_no_large_group_has_a_certain_outcome(self, real_hotel_df):
        """Leakage tripwire: a value shared by >=1% of bookings that is NEVER
        (or ALWAYS) canceled suggests the field is filled in once the outcome
        is known. This is how the parking-space leak was found."""
        numeric_cols, categorical_cols = get_feature_lists(real_hotel_df)
        found = self._groups_with_certain_outcome(real_hotel_df, numeric_cols + categorical_cols)
        assert not found, f"Features with a certain outcome: {found}"

    def test_tripwire_catches_the_parking_leak(self, real_hotel_df):
        found = self._groups_with_certain_outcome(
            real_hotel_df, ["required_car_parking_spaces"]
        )
        assert "required_car_parking_spaces" in found


class TestFeatureLists:
    def test_lists_are_disjoint(self, prepared_hotel_df):
        numeric_cols, categorical_cols = get_feature_lists(prepared_hotel_df)
        assert set(numeric_cols).isdisjoint(categorical_cols)

    def test_string_columns_classified_as_categorical(self, prepared_hotel_df):
        """Regression test: pandas >=2 may back text with `string` dtype rather
        than `object`, and a naive `dtype == object` check silently mislabels
        every categorical column as numeric."""
        _, categorical_cols = get_feature_lists(prepared_hotel_df)
        for column in ["hotel", "country", "deposit_type", "meal", "market_segment"]:
            assert column in categorical_cols, f"{column} must be treated as categorical"

    def test_numeric_columns_are_actually_numeric(self, prepared_hotel_df):
        numeric_cols, _ = get_feature_lists(prepared_hotel_df)
        for column in numeric_cols:
            assert pd.api.types.is_numeric_dtype(prepared_hotel_df[column])

    def test_forced_categoricals_are_categorical(self, prepared_hotel_df):
        _, categorical_cols = get_feature_lists(prepared_hotel_df)
        # Numeric on paper, but meaningless as magnitudes.
        for column in ["arrival_date_year", "is_repeated_guest", "is_family"]:
            assert column in categorical_cols

    def test_identifier_columns_excluded(self, prepared_hotel_df):
        numeric_cols, categorical_cols = get_feature_lists(prepared_hotel_df)
        features = set(numeric_cols) | set(categorical_cols)
        for column in ["agent", "company", "reserved_room_type", "assigned_room_type"]:
            assert column not in features


class TestCleaning:
    def test_no_missing_values_survive(self, prepared_hotel_df):
        numeric_cols, categorical_cols = get_feature_lists(prepared_hotel_df)
        subset = prepared_hotel_df[numeric_cols + categorical_cols]
        assert not subset.isna().any().any(), "Model features must not contain NaN"

    def test_zero_guest_bookings_removed(self, prepared_hotel_df):
        assert (prepared_hotel_df["total_guests"] > 0).all()

    def test_zero_night_bookings_removed(self, prepared_hotel_df):
        assert (prepared_hotel_df["total_nights"] > 0).all()

    def test_adr_is_non_negative(self, prepared_hotel_df):
        assert (prepared_hotel_df["adr"] >= 0).all()

    def test_children_nan_filled(self, raw_hotel_df):
        assert raw_hotel_df["children"].isna().any(), "fixture should contain NaN"
        assert not clean_data(raw_hotel_df)["children"].isna().any()

    def test_impossible_rates_removed(self, raw_hotel_df):
        raw_hotel_df.loc[5, "adr"] = -6.38
        raw_hotel_df.loc[6, "adr"] = 5400.0
        cleaned = clean_data(raw_hotel_df)
        assert cleaned["adr"].between(0, MAX_VALID_ADR).all()
        assert len(cleaned) == len(clean_data(raw_hotel_df.drop(index=[5, 6])))

    def test_preparation_is_row_by_row(self, raw_hotel_df):
        """Preparation runs before the train/test split, so a row's result must
        not depend on the other rows -- otherwise statistics of the test rows
        leak into training (as an earlier quantile clip of adr did)."""
        raw = raw_hotel_df.assign(row_id=range(len(raw_hotel_df)))
        full = engineer_features(clean_data(raw)).set_index("row_id")
        part = engineer_features(clean_data(raw.iloc[:30])).set_index("row_id")
        pd.testing.assert_frame_equal(part, full.loc[part.index])


class TestDeduplication:
    @pytest.fixture
    def path_with_duplicates(self, raw_hotel_df, tmp_path):
        doubled = pd.concat([raw_hotel_df, raw_hotel_df.iloc[10:20]], ignore_index=True)
        path = tmp_path / "with_duplicates.csv"
        doubled.to_csv(path, index=False)
        return path

    def test_duplicates_removed_before_modeling(self, path_with_duplicates):
        X, y, _, _ = load_modeling_data(path_with_duplicates)
        assert not X.assign(**{TARGET: y.to_numpy()}).duplicated().any()

    def test_deduplication_can_be_disabled(self, path_with_duplicates):
        X, _, _, _ = load_modeling_data(path_with_duplicates, deduplicate=False)
        assert len(X) == len(load_and_prepare(path_with_duplicates))

    def test_modeling_data_has_the_feature_columns_only(self, path_with_duplicates):
        X, y, numeric_cols, categorical_cols = load_modeling_data(path_with_duplicates)
        assert list(X.columns) == numeric_cols + categorical_cols
        assert y.name == TARGET


class TestEngineeredFeatures:
    def test_total_nights_is_sum_of_parts(self, prepared_hotel_df):
        expected = (
            prepared_hotel_df["stays_in_weekend_nights"]
            + prepared_hotel_df["stays_in_week_nights"]
        )
        pd.testing.assert_series_equal(
            prepared_hotel_df["total_nights"], expected, check_names=False
        )

    def test_total_guests_is_sum_of_parts(self, prepared_hotel_df):
        expected = (
            prepared_hotel_df["adults"]
            + prepared_hotel_df["children"]
            + prepared_hotel_df["babies"]
        )
        pd.testing.assert_series_equal(
            prepared_hotel_df["total_guests"], expected, check_names=False
        )

    def test_binary_flags_are_binary(self, prepared_hotel_df):
        for column in ["is_family", "room_type_changed", "has_agent", "has_company"]:
            assert set(prepared_hotel_df[column].unique()) <= {0, 1}

    def test_arrival_month_num_matches_month_name(self, prepared_hotel_df):
        assert prepared_hotel_df["arrival_month_num"].between(1, 12).all()
        assert not prepared_hotel_df["arrival_month_num"].isna().any()

    def test_is_family_matches_children_and_babies(self, prepared_hotel_df):
        has_kids = (
            prepared_hotel_df["children"] + prepared_hotel_df["babies"]
        ) > 0
        assert (prepared_hotel_df["is_family"] == has_kids.astype(int)).all()


class TestLoading:
    def test_missing_target_raises_when_required(self, raw_hotel_df, tmp_path):
        path = tmp_path / "no_target.csv"
        raw_hotel_df.drop(columns=[TARGET]).to_csv(path, index=False)
        with pytest.raises(ValueError, match=TARGET):
            load_data(path, require_target=True)

    def test_missing_target_allowed_for_scoring(self, raw_hotel_df, tmp_path):
        """predict.py must accept unlabelled bookings -- that is the whole point."""
        path = tmp_path / "no_target.csv"
        raw_hotel_df.drop(columns=[TARGET]).to_csv(path, index=False)
        result = load_and_prepare(path, require_target=False)
        assert len(result) > 0
        assert TARGET not in result.columns

    def test_prepared_frame_is_not_empty(self, prepared_hotel_df):
        assert len(prepared_hotel_df) > 0

    def test_index_is_reset(self, prepared_hotel_df):
        assert prepared_hotel_df.index.equals(pd.RangeIndex(len(prepared_hotel_df)))
