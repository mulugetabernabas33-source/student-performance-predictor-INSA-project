"""Cached data, model, and evaluation-result loaders shared across pages.

Everything here is wrapped in ``st.cache_resource`` / ``st.cache_data`` so
the (tiny, ~1s) training runs for the comparison scenario and all plot data
only happen once per server process, not on every page view or rerun.
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classify import get_classification_split, run as run_classification  # noqa: E402
from src.data import load_raw_data  # noqa: E402
from src.evaluate import evaluate_model, get_feature_importance  # noqa: E402
from src.evaluate_classification import (  # noqa: E402
    evaluate_classifier,
    get_confusion_matrix,
    get_roc_curve_data,
)
from src.preprocess import clean_data  # noqa: E402
from src.train import run as run_regression  # noqa: E402

MODELS_DIR = PROJECT_ROOT / "models"
REGRESSOR_PATH = MODELS_DIR / "best_model.joblib"
CLASSIFIER_PATH = MODELS_DIR / "best_classifier.joblib"

PASS_THRESHOLD = 10


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    """Load and clean the raw dataset (cached for the session)."""
    return clean_data(load_raw_data())


@st.cache_resource(show_spinner=False)
def load_classifier_pipeline():
    """Load the deployed (Scenario A) classifier pipeline from disk.

    Returns:
        The fitted pipeline, or None if the model file is missing.
    """
    if not CLASSIFIER_PATH.exists():
        return None
    return joblib.load(CLASSIFIER_PATH)


@st.cache_resource(show_spinner=False)
def load_regressor_pipeline():
    """Load the deployed (Scenario A) regressor pipeline from disk.

    Returns:
        The fitted pipeline, or None if the model file is missing.
    """
    if not REGRESSOR_PATH.exists():
        return None
    return joblib.load(REGRESSOR_PATH)


@st.cache_resource(show_spinner=False)
def get_scenario_classification(feature_set: str) -> dict:
    """Train/evaluate a classification scenario end-to-end (cached).

    Args:
        feature_set: ``"early_warning"`` (Scenario A) or ``"full"``
            (Scenario B).

    Returns:
        A dict with the fitted pipeline, leaderboard, test split, metrics,
        confusion matrix, and ROC curve data for this scenario.
    """
    result = run_classification(feature_set=feature_set, save=False)
    metrics = evaluate_classifier(result["pipeline"], result["X_test"], result["y_test"])
    cm = get_confusion_matrix(result["pipeline"], result["X_test"], result["y_test"])
    roc = get_roc_curve_data(result["pipeline"], result["X_test"], result["y_test"])
    return {**result, "metrics": metrics, "confusion_matrix": cm, "roc": roc}


@st.cache_resource(show_spinner=False)
def get_scenario_regression(feature_set: str) -> dict:
    """Train/evaluate a regression scenario end-to-end (cached).

    Args:
        feature_set: ``"early_warning"`` (Scenario A) or ``"full"``
            (Scenario B).

    Returns:
        A dict with the fitted pipeline, leaderboard, test split,
        metrics, and feature importances for this scenario.
    """
    result = run_regression(feature_set=feature_set, save=False)
    metrics = evaluate_model(result["pipeline"], result["X_test"], result["y_test"])
    importance = get_feature_importance(result["pipeline"])
    return {**result, "metrics": metrics, "feature_importance": importance}


def naive_baseline_accuracy(feature_set: str = "early_warning") -> float:
    """Accuracy of an "always predict pass" baseline on the same test split.

    Args:
        feature_set: Which scenario's test split to measure against.

    Returns:
        Fraction of the test split whose true label is "pass".
    """
    df = load_dataset()
    _, _, _, y_test = get_classification_split(df, feature_set)
    return float((y_test == 1).mean())
