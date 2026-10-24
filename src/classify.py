"""Classification models for the binary pass/fail target.

Mirrors src/train.py but for the ``pass`` label (G3 >= 10) instead of the
continuous ``G3`` grade, so the deployed app can show a calibrated
probability, a confusion matrix, and an ROC curve rather than just a
rounded regression output.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src.data import PROJECT_ROOT, load_raw_data
from src.preprocess import build_preprocessor, clean_data

MODELS_DIR = PROJECT_ROOT / "models"
RANDOM_STATE = 42
PASS_LABEL_COLUMN = "pass"


def build_candidate_classifiers() -> dict[str, object]:
    """Return the classifiers compared during model selection.

    Returns:
        A mapping of model name to an unfitted scikit-learn classifier.
    """
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=6, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE
        ),
    }


def get_classification_split(
    df: pd.DataFrame, feature_set: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split a cleaned DataFrame into a train/test set for classification.

    Uses the same ``feature_set`` vocabulary as :mod:`src.preprocess`
    (``"early_warning"`` or ``"full"``) and the same random state as
    :mod:`src.train`, so results are directly comparable across tasks.

    Args:
        df: Cleaned dataset (see :func:`src.preprocess.clean_data`).
        feature_set: ``"early_warning"`` or ``"full"``.

    Returns:
        ``(X_train, X_test, y_train, y_test)`` where ``y`` is the binary
        ``pass`` label.
    """
    from src.preprocess import EARLY_WARNING_FEATURES, FULL_FEATURES

    features = EARLY_WARNING_FEATURES if feature_set == "early_warning" else FULL_FEATURES
    X = df[features].copy()
    y = df[PASS_LABEL_COLUMN].copy()
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)


def cross_validate_classifiers(
    X_train: pd.DataFrame, y_train: pd.Series, feature_set: str, cv: int = 5
) -> pd.DataFrame:
    """Cross-validate each candidate classifier and rank by mean ROC-AUC.

    Args:
        X_train: Training features.
        y_train: Training target (``pass``, 0/1).
        feature_set: ``"early_warning"`` or ``"full"``.
        cv: Number of cross-validation folds.

    Returns:
        A DataFrame with one row per model, sorted by mean ROC-AUC
        descending.
    """
    kfold = KFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    results = []
    for name, model in build_candidate_classifiers().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(feature_set)),
                ("model", model),
            ]
        )
        scores = cross_val_score(
            pipeline, X_train, y_train, cv=kfold, scoring="roc_auc", n_jobs=-1
        )
        results.append(
            {"model": name, "cv_roc_auc_mean": scores.mean(), "cv_roc_auc_std": scores.std()}
        )
    return (
        pd.DataFrame(results)
        .sort_values("cv_roc_auc_mean", ascending=False)
        .reset_index(drop=True)
    )


def train_best_classifier(
    X_train: pd.DataFrame, y_train: pd.Series, feature_set: str, cv: int = 5
) -> tuple[Pipeline, pd.DataFrame]:
    """Cross-validate candidates and fit the best one on the full training set.

    Args:
        X_train: Training features.
        y_train: Training target (``pass``, 0/1).
        feature_set: ``"early_warning"`` or ``"full"``.
        cv: Number of cross-validation folds.

    Returns:
        A tuple of (fitted best pipeline, cross-validation leaderboard).
    """
    leaderboard = cross_validate_classifiers(X_train, y_train, feature_set, cv=cv)
    best_name = leaderboard.iloc[0]["model"]
    best_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(feature_set)),
            ("model", build_candidate_classifiers()[best_name]),
        ]
    )
    best_pipeline.fit(X_train, y_train)
    return best_pipeline, leaderboard


def run(feature_set: str = "early_warning", save: bool = True) -> dict:
    """End-to-end classification training run.

    Args:
        feature_set: ``"early_warning"`` or ``"full"``.
        save: If True, persist the fitted pipeline (early_warning only)
            and a metrics summary to the ``models/`` directory.

    Returns:
        A dict with the fitted pipeline, leaderboard, and train/test split.
    """
    raw = load_raw_data()
    cleaned = clean_data(raw)
    X_train, X_test, y_train, y_test = get_classification_split(cleaned, feature_set)

    pipeline, leaderboard = train_best_classifier(X_train, y_train, feature_set)

    if save and feature_set == "early_warning":
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        import joblib

        joblib.dump(pipeline, MODELS_DIR / "best_classifier.joblib")

        metadata = {
            "feature_set": feature_set,
            "best_model": leaderboard.iloc[0]["model"],
            "cv_roc_auc_mean": leaderboard.iloc[0]["cv_roc_auc_mean"],
            "leaderboard": leaderboard.to_dict(orient="records"),
        }
        with open(MODELS_DIR / "classifier_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    return {
        "pipeline": pipeline,
        "leaderboard": leaderboard,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the pass/fail classifier.")
    parser.add_argument(
        "--feature-set",
        choices=["early_warning", "full"],
        default="early_warning",
        help="Feature set to train on (default: early_warning).",
    )
    args = parser.parse_args()

    result = run(feature_set=args.feature_set)
    print(result["leaderboard"])
    print(f"Best model: {result['leaderboard'].iloc[0]['model']}")


if __name__ == "__main__":
    main()
