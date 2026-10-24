"""Data loading utilities for the UCI Student Performance dataset."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

MATH_CSV_PATH = RAW_DATA_DIR / "student-mat.csv"
PORTUGUESE_CSV_PATH = RAW_DATA_DIR / "student-por.csv"


def load_raw_data(path: Path = MATH_CSV_PATH) -> pd.DataFrame:
    """Load a raw UCI Student Performance CSV file.

    The source files are semicolon-separated and ship with quoted string
    values, so both quirks are handled here.

    Args:
        path: Path to a ``student-mat.csv`` or ``student-por.csv`` file.

    Returns:
        The raw dataset as a DataFrame, unchanged except for parsing.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. See data/raw/student.txt or the "
            "README for download instructions."
        )
    return pd.read_csv(path, sep=";")


def load_processed_data(path: Path = PROCESSED_DATA_DIR / "student_clean.csv") -> pd.DataFrame:
    """Load the cleaned dataset produced by :mod:`src.preprocess`.

    Args:
        path: Path to the processed CSV file.

    Returns:
        The cleaned dataset as a DataFrame.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {path}. Run `python -m "
            "src.preprocess` first."
        )
    return pd.read_csv(path)
