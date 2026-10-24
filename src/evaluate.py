"""Evaluation utilities for the fitted student grade prediction pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src.preprocess import PASS_THRESHOLD


def evaluate_model(
    pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:
    """Compute regression and derived pass/fail metrics on a held-out set.

    Args:
        pipeline: A fitted scikit-learn pipeline.
        X_test: Held-out features.
        y_test: Held-out true final grades (G3).

    Returns:
        A dict with ``rmse``, ``mae``, ``r2``, and ``pass_fail_accuracy``
        (accuracy of the pass/fail label derived from rounding and
        thresholding the predicted grade at ``PASS_THRESHOLD``).
    """
    y_pred = pipeline.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    y_test_pass = (y_test >= PASS_THRESHOLD).astype(int)
    y_pred_pass = (np.round(y_pred) >= PASS_THRESHOLD).astype(int)
    pass_fail_accuracy = float((y_test_pass.values == y_pred_pass).mean())

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pass_fail_accuracy": pass_fail_accuracy,
    }


def get_predictions(pipeline: Pipeline, X_test: pd.DataFrame) -> np.ndarray:
    """Return raw model predictions for a feature matrix.

    Args:
        pipeline: A fitted scikit-learn pipeline.
        X_test: Features to predict on.

    Returns:
        Array of predicted final grades.
    """
    return pipeline.predict(X_test)


def get_feature_importance(pipeline: Pipeline) -> pd.DataFrame | None:
    """Extract feature importances from a fitted tree-based pipeline.

    Args:
        pipeline: A fitted pipeline whose final step exposes
            ``feature_importances_`` (e.g. RandomForest, GradientBoosting).
            Returns None for models without this attribute (e.g. linear
            models).

    Returns:
        A DataFrame with ``feature`` and ``importance`` columns sorted
        descending, or None if the model has no importances.
    """
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return None

    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_

    return (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
