"""Common data loading utilities for all pipelines."""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import get_data_file


def load_and_clean_data(data_file: Path = None) -> pd.DataFrame:
    """Load and clean dataset with proper CSV parsing.
    
    Args:
        data_file: Path to CSV file (uses DATA_FILE env if not specified)
        
    Returns:
        Cleaned DataFrame with 'text' and 'label' columns
        
    Raises:
        ValueError: If required columns are missing
    """
    if data_file is None:
        data_file = get_data_file()
    
    # Load data
    try:
        data = pd.read_csv(data_file, encoding="utf-8", quotechar='"', skipinitialspace=True)
    except Exception:
        data = pd.read_csv(
            data_file,
            encoding="utf-8",
            quotechar='"',
            skipinitialspace=True,
            on_bad_lines="skip"
        )
    
    # Validate columns
    if "text" not in data.columns or "label" not in data.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns")
    
    # Clean text
    data["text"] = data["text"].astype(str)
    data["text"] = data["text"].str.replace("\n", " ").str.replace("\r", " ").str.strip()
    data = data[data["text"] != ""]
    
    # Clean labels
    data["label"] = pd.to_numeric(data["label"], errors="coerce")
    data = data.dropna(subset=["label"])
    data["label"] = data["label"].astype(int)
    
    # Handle {1,2} labels (map to {1,0})
    if set(data["label"].unique()).issubset({1, 2}):
        data["label"] = data["label"].replace({2: 0})
    
    # Keep only valid {0, 1} labels
    data = data[data["label"].isin([0, 1])]
    
    return data


def create_train_test_split(
    data: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create stratified train/test split.
    
    Args:
        data: Input DataFrame with 'text' and 'label' columns
        test_size: Test set fraction
        random_state: Random seed
        
    Returns:
        Tuple of (texts_train, texts_test, labels_train, labels_test)
    """
    texts = data["text"].astype(str).to_numpy(dtype=object)
    labels = pd.to_numeric(data["label"], errors="coerce").to_numpy(dtype=np.int64)
    
    texts_train, texts_test, labels_train, labels_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
        shuffle=True,
    )
    
    return texts_train, texts_test, labels_train, labels_test


def analyze_label_distribution(labels: np.ndarray, title: str = "Label Distribution") -> dict:
    """Analyze and log label distribution statistics.
    
    Args:
        labels: Binary label array
        title: Title for logging
        
    Returns:
        Dict with distribution statistics
    """
    total = len(labels)
    count_0 = np.sum(labels == 0)
    count_1 = np.sum(labels == 1)
    
    return {
        "total_samples": total,
        "count_0": count_0,
        "count_1": count_1,
        "ratio_0": round(count_0 / total * 100, 1),
        "ratio_1": round(count_1 / total * 100, 1),
        "imbalance_ratio": round(max(count_0, count_1) / min(count_0, count_1), 2),
    }
