"""Feature engineering and preprocessing for the student performance dataset."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import PROCESSED_DATA_DIR, load_raw_data

TARGET = "G3"
PASS_THRESHOLD = 10

# Features available at the start of the term, before any grades exist.
# Used to build an "early-warning" model that can flag at-risk students
# before the first graded assessment.
EARLY_WARNING_FEATURES = [
    "school",
    "sex",
    "age",
    "address",
    "famsize",
    "Pstatus",
    "Medu",
    "Fedu",
    "Mjob",
    "Fjob",
    "reason",
    "guardian",
    "traveltime",
    "studytime",
    "failures",
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "nursery",
    "higher",
    "internet",
    "romantic",
    "famrel",
    "freetime",
    "goout",
    "Dalc",
    "Walc",
    "health",
    "absences",
]

# Full feature set additionally includes the first and second period grades.
FULL_FEATURES = EARLY_WARNING_FEATURES + ["G1", "G2"]

NUMERIC_FEATURES = [
    "age",
    "Medu",
    "Fedu",
    "traveltime",
    "studytime",
    "failures",
    "famrel",
    "freetime",
    "goout",
    "Dalc",
    "Walc",
    "health",
    "absences",
]
CATEGORICAL_FEATURES = [
    "school",
    "sex",
    "address",
    "famsize",
    "Pstatus",
    "Mjob",
    "Fjob",
    "reason",
    "guardian",
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "nursery",
    "higher",
    "internet",
    "romantic",
]


def add_pass_fail_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add a binary ``pass`` column derived from the final grade.

    Args:
        df: DataFrame containing a ``G3`` column (final grade, 0-20).

    Returns:
        A copy of ``df`` with an added ``pass`` column (1 if G3 >= 10).
    """
    df = df.copy()
    df["pass"] = (df[TARGET] >= PASS_THRESHOLD).astype(int)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and lightly clean the raw dataset.

    The UCI source file has no missing values or duplicate rows, but this
    function defensively drops any that appear and derives the ``pass``
    column used for reporting classification-style metrics.

    Args:
        df: Raw dataset as loaded by :func:`src.data.load_raw_data`.

    Returns:
        A cleaned copy of the dataset.
    """
    df = df.drop_duplicates().dropna().reset_index(drop=True)
    return add_pass_fail_column(df)


def get_feature_target_split(
    df: pd.DataFrame, feature_set: str = "early_warning"
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a cleaned DataFrame into model inputs ``X`` and target ``y``.

    Args:
        df: Cleaned dataset (see :func:`clean_data`).
        feature_set: Either ``"early_warning"`` (demographic/behavioral
            features only) or ``"full"`` (also includes G1 and G2).

    Returns:
        A tuple ``(X, y)`` where ``y`` is the final grade ``G3``.
    """
    if feature_set == "early_warning":
        features = EARLY_WARNING_FEATURES
    elif feature_set == "full":
        features = FULL_FEATURES
    else:
        raise ValueError(f"Unknown feature_set: {feature_set!r}")
    return df[features].copy(), df[TARGET].copy()


def build_preprocessor(feature_set: str = "early_warning") -> ColumnTransformer:
    """Build the scikit-learn preprocessing pipeline for a feature set.

    Numeric columns are standard-scaled and categorical columns are
    one-hot encoded.

    Args:
        feature_set: Either ``"early_warning"`` or ``"full"``; only
            affects whether ``G1``/``G2`` are included as numeric features.

    Returns:
        An unfitted :class:`~sklearn.compose.ColumnTransformer`.
    """
    numeric_features = list(NUMERIC_FEATURES)
    if feature_set == "full":
        numeric_features += ["G1", "G2"]

    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def save_processed_data(
    df: pd.DataFrame, path: Path = PROCESSED_DATA_DIR / "student_clean.csv"
) -> Path:
    """Persist the cleaned dataset to ``data/processed``.

    Args:
        df: Cleaned dataset to save.
        path: Destination CSV path.

    Returns:
        The path the data was written to.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def main() -> None:
    """Load the raw Math dataset, clean it, and save it to data/processed."""
    raw = load_raw_data()
    cleaned = clean_data(raw)
    output_path = save_processed_data(cleaned)
    print(f"Saved cleaned dataset with shape {cleaned.shape} to {output_path}")


if __name__ == "__main__":
    main()
