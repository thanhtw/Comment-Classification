"""Common data loading utilities for all pipelines.

The key function is ``get_canonical_split()`` which guarantees that every
pipeline (ML, DL, Transformer, LLM) operates on the **exact same** cleaned
dataset and held-out test split.  This ensures confusion-matrix sample counts
are identical across all models.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import get_data_file, get_project_root

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level cache so the split is computed at most once per process.
# ---------------------------------------------------------------------------
_CANONICAL_CACHE: Optional[Dict] = None

HOLDOUT_TEST_SIZE = 0.2
CANONICAL_RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Canonical cleaning (single source of truth)
# ---------------------------------------------------------------------------

def load_and_clean_data(data_file: Path = None) -> pd.DataFrame:
    """Load and clean dataset with proper CSV parsing.

    This is the **single** cleaning pipeline used by every model.  All four
    model pipelines must route through this function so that the resulting
    DataFrame (and therefore any downstream split) is identical.

    Args:
        data_file: Path to CSV file (uses DATA_FILE env if not specified)

    Returns:
        Cleaned DataFrame with ``text`` and ``label`` columns.

    Raises:
        ValueError: If required columns are missing.
    """
    if data_file is None:
        data_file = get_data_file()

    data_file = Path(data_file)
    if not data_file.is_absolute():
        data_file = (get_project_root() / data_file).resolve()

    # Load CSV ---------------------------------------------------------
    try:
        data = pd.read_csv(data_file, encoding="utf-8", quotechar='"', skipinitialspace=True)
    except Exception:
        data = pd.read_csv(
            data_file,
            encoding="utf-8",
            quotechar='"',
            skipinitialspace=True,
            on_bad_lines="skip",
        )

    # Validate columns -------------------------------------------------
    if "text" not in data.columns or "label" not in data.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns")

    # Clean text -------------------------------------------------------
    data["text"] = data["text"].astype(str)
    data["text"] = (
        data["text"]
        .str.replace("\n", " ", regex=False)
        .str.replace("\r", " ", regex=False)
        .str.strip()
    )
    data = data[data["text"] != ""]

    # Clean labels -----------------------------------------------------
    data["label"] = pd.to_numeric(data["label"], errors="coerce")
    data = data.dropna(subset=["label"])
    data["label"] = data["label"].astype(int)

    # Handle {1, 2} labels -> {1, 0}
    if set(data["label"].unique()).issubset({1, 2}):
        data["label"] = data["label"].replace({2: 0})

    # Keep only valid binary labels
    data = data[data["label"].isin([0, 1])]

    data = data.reset_index(drop=True)
    return data


# ---------------------------------------------------------------------------
# Canonical train / test split
# ---------------------------------------------------------------------------

def create_train_test_split(
    data: pd.DataFrame,
    test_size: float = HOLDOUT_TEST_SIZE,
    random_state: int = CANONICAL_RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a stratified train/test split from a cleaned DataFrame.

    Args:
        data: DataFrame with ``text`` and ``label`` columns.
        test_size: Fraction held out for testing.
        random_state: Random seed for reproducibility.

    Returns:
        (texts_train, texts_test, labels_train, labels_test) as NumPy arrays.
    """
    texts = data["text"].astype(str).to_numpy(dtype=object)
    labels = pd.to_numeric(data["label"], errors="coerce").to_numpy(dtype=np.int64)

    return train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
        shuffle=True,
    )


def get_canonical_split(
    data_file: Path = None,
) -> Dict:
    """Return the **single** canonical train/test split used by all pipelines.

    The first call loads and cleans the data, creates the split, and caches
    the result.  Subsequent calls return the cached copy so every pipeline
    in the same process is guaranteed to receive the same arrays.

    Returns a dict with keys:

    * ``data``           – the full cleaned DataFrame
    * ``texts_train``    – training texts   (np.ndarray, dtype=object)
    * ``texts_test``     – test texts       (np.ndarray, dtype=object)
    * ``labels_train``   – training labels  (np.ndarray, dtype=int64)
    * ``labels_test``    – test labels      (np.ndarray, dtype=int64)
    * ``train_indices``  – row indices in ``data`` for the training set
    * ``test_indices``   – row indices in ``data`` for the test set
    """
    global _CANONICAL_CACHE

    if _CANONICAL_CACHE is not None:
        return _CANONICAL_CACHE

    data = load_and_clean_data(data_file)

    texts = data["text"].astype(str).to_numpy(dtype=object)
    labels = pd.to_numeric(data["label"], errors="coerce").to_numpy(dtype=np.int64)
    indices = np.arange(len(data))

    (
        texts_train, texts_test,
        labels_train, labels_test,
        idx_train, idx_test,
    ) = train_test_split(
        texts,
        labels,
        indices,
        test_size=HOLDOUT_TEST_SIZE,
        random_state=CANONICAL_RANDOM_STATE,
        stratify=labels,
        shuffle=True,
    )

    _CANONICAL_CACHE = {
        "data": data,
        "texts_train": texts_train,
        "texts_test": texts_test,
        "labels_train": labels_train,
        "labels_test": labels_test,
        "train_indices": idx_train,
        "test_indices": idx_test,
    }

    n_train = len(labels_train)
    n_test = len(labels_test)
    logger.info(
        f"Canonical split created: {n_train} train / {n_test} test "
        f"(total {n_train + n_test})"
    )

    return _CANONICAL_CACHE


def reset_canonical_cache() -> None:
    """Clear the cached split (useful for testing)."""
    global _CANONICAL_CACHE
    _CANONICAL_CACHE = None


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def analyze_label_distribution(labels: np.ndarray, title: str = "Label Distribution") -> dict:
    """Analyze and log label distribution statistics."""
    total = len(labels)
    count_0 = int(np.sum(labels == 0))
    count_1 = int(np.sum(labels == 1))

    return {
        "total_samples": total,
        "count_0": count_0,
        "count_1": count_1,
        "ratio_0": round(count_0 / total * 100, 1) if total else 0,
        "ratio_1": round(count_1 / total * 100, 1) if total else 0,
        "imbalance_ratio": round(max(count_0, count_1) / max(min(count_0, count_1), 1), 2),
    }
