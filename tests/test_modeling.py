"""Tests for modeling.py -- the shared models and evaluation protocol."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.svm import SVC

from modeling import (
    HOTEL_METRICS,
    MODEL_NAMES,
    SubsampledClassifier,
    build_pipeline,
    build_preprocessor,
    describe_distribution,
    load_params,
    out_of_fold_scores,
    positive_scores,
    pretty_feature_names,
    repeated_cv,
    save_params,
    search_space,
)


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame({
        "a": rng.normal(size=n),
        "b": rng.normal(size=n),
        "c": rng.choice(["x", "y", "z"], n),
    })
    y = pd.Series(((X["a"] + rng.normal(scale=0.5, size=n)) > 0.8).astype(int), name="target")
    return X, y


class TestSubsampledClassifier:
    def test_stratified_keeps_size_and_class_ratio(self):
        y = np.array([1] * 300 + [0] * 700)
        keep = SubsampledClassifier(SVC(), max_samples=100, strategy="stratified")._sample_indices(y)
        assert len(keep) == 100
        assert y[keep].mean() == pytest.approx(0.3, abs=0.01)

    def test_undersample_keeps_every_positive(self):
        y = np.array([1] * 20 + [0] * 980)
        keep = SubsampledClassifier(SVC(), max_samples=100, strategy="undersample")._sample_indices(y)
        assert len(keep) == 100
        assert y[keep].sum() == 20

    def test_small_data_is_used_whole(self):
        y = np.array([0, 1] * 10)
        keep = SubsampledClassifier(SVC(), max_samples=100)._sample_indices(y)
        np.testing.assert_array_equal(keep, np.arange(20))

    def test_fits_on_subsample_but_predicts_every_row(self, toy_data):
        X, y = toy_data
        clf = SubsampledClassifier(SVC(), max_samples=100).fit(X[["a", "b"]], y)
        assert clf.n_samples_fit_ == 100
        assert len(clf.predict(X[["a", "b"]])) == len(X)

    def test_svc_gives_scores_not_probabilities(self):
        clf = SubsampledClassifier(SVC())
        assert hasattr(clf, "decision_function")
        assert not hasattr(clf, "predict_proba")

    def test_inner_parameters_are_tunable(self):
        clf = SubsampledClassifier(SVC(C=1.0)).set_params(estimator__C=5.0)
        assert clone(clf).get_params()["estimator__C"] == 5.0


class TestPipelines:
    @pytest.mark.parametrize("task", ["hotel", "card"])
    @pytest.mark.parametrize("name", MODEL_NAMES)
    def test_every_model_trains_and_scores(self, toy_data, name, task):
        X, y = toy_data
        pipeline = build_pipeline(name, task, ["a", "b"], ["c"], y).fit(X, y)
        scores = positive_scores(pipeline, X)
        assert scores.shape == (len(X),)
        assert np.isfinite(scores).all()

    @pytest.mark.parametrize("task", ["hotel", "card"])
    @pytest.mark.parametrize("name", MODEL_NAMES)
    def test_search_space_names_real_parameters(self, toy_data, name, task):
        """A typo in a search-space key would only surface deep into a long run."""
        X, y = toy_data
        available = build_pipeline(name, task, ["a", "b"], ["c"], y).get_params()
        for key in search_space(name, task, y):
            assert key in available, f"{key} is not a parameter of the {name} pipeline"

    def test_rare_and_unseen_categories_share_one_column(self):
        train = pd.DataFrame({"a": np.arange(1000.0),
                              "c": ["x"] * 600 + ["y"] * 395 + ["rare"] * 5})
        preprocessor = build_preprocessor(["a"], ["c"]).fit(train)
        names = list(preprocessor.get_feature_names_out())
        assert "cat__c_infrequent_sklearn" in names
        unseen = preprocessor.transform(pd.DataFrame({"a": [1.0], "c": ["never seen"]}))
        unseen = unseen.toarray() if hasattr(unseen, "toarray") else unseen
        assert unseen[0, names.index("cat__c_infrequent_sklearn")] == 1


class TestRepeatedCV:
    def test_out_of_fold_scores_cover_every_row(self, toy_data):
        X, y = toy_data
        pipeline = build_pipeline("logistic_regression", "hotel", ["a", "b"], ["c"], y)
        folds, oof = repeated_cv(pipeline, X, y, HOTEL_METRICS, n_splits=5, n_repeats=2)
        assert len(folds) == 10
        assert set(folds.columns) == set(HOTEL_METRICS)
        assert not np.isnan(oof).any()

    def test_out_of_fold_helper_reproduces_repeated_cv(self, toy_data):
        """explain.py must recompute exactly the scores card_fraud.py used."""
        X, y = toy_data
        pipeline = build_pipeline("logistic_regression", "card", ["a", "b"], ["c"], y)
        _, oof = repeated_cv(pipeline, X, y, HOTEL_METRICS, n_repeats=2)
        np.testing.assert_allclose(out_of_fold_scores(pipeline, X, y), oof)


class TestHelpers:
    def test_pretty_feature_names(self):
        names = ["num__lead_time", "cat__country_PRT", "cat__deposit_type_Non Refund",
                 "cat__country_infrequent_sklearn"]
        assert pretty_feature_names(names, ["country", "deposit_type"]) == [
            "lead_time", "country = PRT", "deposit_type = Non Refund", "country = ostale (rijetke)",
        ]

    def test_describe_distribution(self):
        assert describe_distribution(randint(100, 400)) == "cijeli broj iz [100, 399]"
        assert describe_distribution(uniform(0.6, 0.4)) == "uniformno iz [0,6; 1]"
        assert describe_distribution(loguniform(0.001, 100)) == "log-uniformno iz [0,001; 100]"
        assert describe_distribution([None, "balanced"]) == "{None; balanced}"

    def test_params_round_trip_with_numpy_types(self, tmp_path):
        path = tmp_path / "params.json"
        save_params(path, {"xgboost": {"max_depth": np.int64(6),
                                       "learning_rate": np.float64(0.1),
                                       "class_weight": None}})
        loaded = load_params(path)
        assert loaded == {"xgboost": {"max_depth": 6, "learning_rate": 0.1, "class_weight": None}}
        assert isinstance(loaded["xgboost"]["max_depth"], int)

    def test_missing_params_file_means_defaults(self, tmp_path):
        assert load_params(tmp_path / "missing.json") == {}
