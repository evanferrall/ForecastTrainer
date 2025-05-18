import re
import numpy as np
import pandas as pd # Added as make_unique_cols might operate on pd.Index/Series in future, or type hints

# Constants
VALID_STATUSES = {"normal", "paid", "confirmed"}
EPSILON = 1e-6  # For logit transform to avoid log(0) or division by zero

# Module-level constants from the roadmap
ONLINE_GAME_INDICATORS = ("online", "virtual", "remote")
LEAD_TIME_BINS = [0, 1, 2, 4, 8, 15, 31, float('inf')]
LEAD_TIME_LABELS = ['0d', '1d', '2_3d', '4_7d', '8_14d', '15_30d', 'gt_30d']


def logit_transform(x: pd.Series) -> pd.Series:
    """Applies a logit transformation to a pandas Series."""
    return np.log((x + EPSILON) / (1 - x + EPSILON))


def _snake(s: str) -> str:
    """Converts a string to snake_case, removing content in parentheses."""
    s = re.sub(r"\(.*?\)", "", s).strip().lower()
    return re.sub(r"[^\w]+", "_", s)


def make_unique_cols(columns: list[str]) -> list[str]:
    """Makes column names unique by appending a suffix if needed."""
    seen: dict[str, int] = {}
    new_columns: list[str] = []
    for col in columns:
        if col not in seen:
            seen[col] = 0
            new_columns.append(col)
        else:
            seen[col] += 1
            new_columns.append(f"{col}_{seen[col]}")
    return new_columns 