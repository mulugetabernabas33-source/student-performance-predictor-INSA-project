"""Smoke tests for the data, preprocessing, and training pipeline."""

import numpy as np

from src.classify import get_classification_split, train_best_classifier
from src.data import load_raw_data
from src.preprocess import clean_data, get_feature_target_split
from src.train import train_best_model


def test_data_loads():
    """The raw dataset loads with the expected shape and target column."""
    df = load_raw_data()
    assert df.shape[0] > 0
    assert "G3" in df.columns


def test_pipeline_fits():
    """The preprocessing + model pipeline fits without error on real data."""
    df = clean_data(load_raw_data())
    X, y = get_feature_target_split(df, feature_set="early_warning")
    pipeline, leaderboard = train_best_model(X, y, feature_set="early_warning", cv=3)
    assert len(leaderboard) == 4
    assert hasattr(pipeline.named_steps["model"], "predict")


def test_model_predicts_valid_output():
    """Predictions are finite numbers within a plausible grade range."""
    df = clean_data(load_raw_data())
    X, y = get_feature_target_split(df, feature_set="early_warning")
    pipeline, _ = train_best_model(X, y, feature_set="early_warning", cv=3)

    predictions = pipeline.predict(X.head(10))
    assert predictions.shape == (10,)
    assert np.all(np.isfinite(predictions))
    assert np.all(predictions > -5) and np.all(predictions < 25)


def test_classifier_predicts_valid_probabilities():
    """The pass/fail classifier outputs well-formed probabilities."""
    df = clean_data(load_raw_data())
    X_train, X_test, y_train, _ = get_classification_split(df, feature_set="early_warning")
    pipeline, leaderboard = train_best_classifier(X_train, y_train, feature_set="early_warning", cv=3)

    assert len(leaderboard) == 3
    proba = pipeline.predict_proba(X_test.head(10))[:, 1]
    assert proba.shape == (10,)
    assert np.all((proba >= 0) & (proba <= 1))
