"""Evaluation and explainability utilities for the pass/fail classifier."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

FAIL_CLASS = 0
PASS_CLASS = 1


def evaluate_classifier(
    pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:
    """Compute classification metrics on a held-out set.

    Args:
        pipeline: A fitted scikit-learn classification pipeline exposing
            ``predict_proba``.
        X_test: Held-out features.
        y_test: Held-out true pass/fail labels (1 = pass, 0 = fail).

    Returns:
        A dict with accuracy, precision/recall/f1 for the pass class,
        recall for the fail class (the metric that matters most for an
        early-warning tool, since missing an at-risk student is the
        costly error), and ROC-AUC.
    """
    y_proba = pipeline.predict_proba(X_test)[:, PASS_CLASS]
    y_pred = (y_proba >= 0.5).astype(int)

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_pass": float(precision_score(y_test, y_pred, pos_label=PASS_CLASS)),
        "recall_pass": float(recall_score(y_test, y_pred, pos_label=PASS_CLASS)),
        "f1_pass": float(f1_score(y_test, y_pred, pos_label=PASS_CLASS)),
        "recall_fail": float(recall_score(y_test, y_pred, pos_label=FAIL_CLASS)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }


def get_confusion_matrix(
    pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> np.ndarray:
    """Return a 2x2 confusion matrix ordered [[fail, fail], [pass, pass]].

    Args:
        pipeline: A fitted classification pipeline.
        X_test: Held-out features.
        y_test: Held-out true pass/fail labels.

    Returns:
        A 2x2 array: rows are true labels, columns are predicted labels,
        both ordered (fail, pass).
    """
    y_proba = pipeline.predict_proba(X_test)[:, PASS_CLASS]
    y_pred = (y_proba >= 0.5).astype(int)
    return confusion_matrix(y_test, y_pred, labels=[FAIL_CLASS, PASS_CLASS])


@dataclass
class RocCurveData:
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray
    auc: float


def get_roc_curve_data(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> RocCurveData:
    """Compute ROC curve points for plotting.

    Args:
        pipeline: A fitted classification pipeline.
        X_test: Held-out features.
        y_test: Held-out true pass/fail labels.

    Returns:
        A :class:`RocCurveData` with false/true positive rates and AUC.
    """
    y_proba = pipeline.predict_proba(X_test)[:, PASS_CLASS]
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    return RocCurveData(fpr=fpr, tpr=tpr, thresholds=thresholds, auc=float(auc))


def predict_proba_single(pipeline: Pipeline, X_row: pd.DataFrame) -> float:
    """Return the predicted probability of passing for a single student.

    Args:
        pipeline: A fitted classification pipeline.
        X_row: A single-row DataFrame of early-warning features.

    Returns:
        Probability of the ``pass`` class, in [0, 1].
    """
    return float(pipeline.predict_proba(X_row)[0, PASS_CLASS])


def explain_prediction(pipeline: Pipeline, X_row: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    """Explain a single prediction with SHAP values from the tree model.

    One-hot-encoded columns are collapsed back to their original feature
    (summing their SHAP contributions) so the explanation reads in terms
    of the form fields a user actually filled in, e.g. "Mjob" rather than
    "Mjob_teacher".

    Args:
        pipeline: A fitted pipeline whose final step is a tree-based
            classifier (RandomForest or GradientBoosting) supported by
            :class:`shap.TreeExplainer`.
        X_row: A single-row DataFrame of raw (pre-preprocessing) features.
        top_n: Maximum number of features to return, ranked by the
            absolute size of their contribution.

    Returns:
        A DataFrame with columns ``feature``, ``shap_value`` (positive
        pushes toward "pass", negative toward "fail"), and ``abs_value``,
        sorted by ``abs_value`` descending.
    """
    import shap

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    X_transformed = preprocessor.transform(X_row)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    transformed_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(model)
    raw_shap = explainer.shap_values(X_transformed)

    # Newer SHAP versions return an array shaped (n_samples, n_features, n_classes)
    # for multiclass/binary trees; older ones return a list of per-class arrays.
    if isinstance(raw_shap, list):
        values = np.asarray(raw_shap[PASS_CLASS])[0]
    elif raw_shap.ndim == 3:
        values = raw_shap[0, :, PASS_CLASS]
    else:
        values = raw_shap[0]

    # Collapse one-hot columns ("categorical__Mjob_teacher") back to their
    # source feature ("Mjob") by summing contributions.
    categorical_columns = preprocessor.named_transformers_["categorical"].feature_names_in_
    contributions: dict[str, float] = {}
    for name, value in zip(transformed_names, values):
        base_name = name.split("__", 1)[-1]
        for cat_col in categorical_columns:
            if base_name.startswith(f"{cat_col}_"):
                base_name = cat_col
                break
        contributions[base_name] = contributions.get(base_name, 0.0) + float(value)

    result = pd.DataFrame(
        [{"feature": k, "shap_value": v, "abs_value": abs(v)} for k, v in contributions.items()]
    )
    return result.sort_values("abs_value", ascending=False).head(top_n).reset_index(drop=True)
