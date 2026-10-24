"""Model training and selection for predicting student final grades (G3)."""

import argparse
import json

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src.data import PROJECT_ROOT, load_raw_data
from src.preprocess import build_preprocessor, clean_data, get_feature_target_split

MODELS_DIR = PROJECT_ROOT / "models"
RANDOM_STATE = 42


def build_candidate_models() -> dict[str, object]:
    """Return the regressors compared during model selection.

    Returns:
        A mapping of model name to an unfitted scikit-learn estimator.
    """
    return {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "random_forest": RandomForestRegressor(
            n_estimators=200, max_depth=8, random_state=RANDOM_STATE
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE
        ),
    }


def cross_validate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_set: str,
    cv: int = 5,
) -> pd.DataFrame:
    """Cross-validate each candidate model and rank by mean R^2.

    Args:
        X_train: Training features.
        y_train: Training target (G3).
        feature_set: ``"early_warning"`` or ``"full"`` (passed to the
            preprocessor so numeric columns match).
        cv: Number of cross-validation folds.

    Returns:
        A DataFrame with one row per model, sorted by mean R^2 descending.
    """
    kfold = KFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    results = []
    for name, model in build_candidate_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(feature_set)),
                ("model", model),
            ]
        )
        scores = cross_val_score(
            pipeline, X_train, y_train, cv=kfold, scoring="r2", n_jobs=-1
        )
        results.append({"model": name, "cv_r2_mean": scores.mean(), "cv_r2_std": scores.std()})
    return pd.DataFrame(results).sort_values("cv_r2_mean", ascending=False).reset_index(drop=True)


def train_best_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_set: str,
    cv: int = 5,
) -> tuple[Pipeline, pd.DataFrame]:
    """Cross-validate candidates and fit the best one on the full training set.

    Args:
        X_train: Training features.
        y_train: Training target (G3).
        feature_set: ``"early_warning"`` or ``"full"``.
        cv: Number of cross-validation folds.

    Returns:
        A tuple of (fitted best pipeline, cross-validation leaderboard).
    """
    leaderboard = cross_validate_models(X_train, y_train, feature_set, cv=cv)
    best_name = leaderboard.iloc[0]["model"]
    best_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(feature_set)),
            ("model", build_candidate_models()[best_name]),
        ]
    )
    best_pipeline.fit(X_train, y_train)
    return best_pipeline, leaderboard


def run(feature_set: str = "early_warning", save: bool = True) -> dict:
    """End-to-end training run: load, split, select, fit, and optionally save.

    Args:
        feature_set: ``"early_warning"`` or ``"full"``.
        save: If True, persist the fitted pipeline and a metrics summary
            to the ``models/`` directory.

    Returns:
        A dict with the fitted pipeline, leaderboard, and train/test split
        (useful for the caller to run further evaluation).
    """
    raw = load_raw_data()
    cleaned = clean_data(raw)
    X, y = get_feature_target_split(cleaned, feature_set=feature_set)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    pipeline, leaderboard = train_best_model(X_train, y_train, feature_set)

    if save:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        import joblib

        model_path = MODELS_DIR / "best_model.joblib"
        joblib.dump(pipeline, model_path)

        metadata = {
            "feature_set": feature_set,
            "best_model": leaderboard.iloc[0]["model"],
            "cv_r2_mean": leaderboard.iloc[0]["cv_r2_mean"],
            "leaderboard": leaderboard.to_dict(orient="records"),
        }
        with open(MODELS_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
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
    parser = argparse.ArgumentParser(description="Train the student grade predictor.")
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
