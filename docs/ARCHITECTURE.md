"""
ARCHITECTURE.md - System design and architecture overview
"""

# Project Architecture

## Overview

This project implements binary comment classification using four complementary approaches:
1. **Classical ML**: SVM, Naive Bayes, Random Forest
2. **Deep Learning**: LSTM, BiLSTM
3. **Transformers**: BERT, RoBERTa
4. **LLM-based**: Zero-shot and few-shot prompting via Groq

All pipelines follow a consistent 10-fold cross-validation strategy with standardized best-fold selection.

## Data Flow

```
Data (CSV)
    ↓
load_and_clean_data()
    ↓
create_train_test_split()
    ↓
[10-fold stratified CV]
    ├→ Fold 1: Train | Validate
    ├→ Fold 2: Train | Validate
    ...
    └→ Fold 10: Train | Validate
    ↓
[Select best fold by F1 score]
    ↓
[Generate metrics, predictions, reports]
    ↓
results/pipeline_name/
├── artifacts/
│   ├── model_best_fold_predictions.csv
│   ├── model_cv_predictions.csv
│   ├── metrics.json
│   └── ...
├── figures/
│   ├── confusion_matrix.png
│   ├── performance_comparison.png
│   └── ...
├── models/
│   └── trained_models.pkl
└── reports/
    └── analysis.md
```

## Pipeline Execution

Each pipeline is independent and follows this pattern:

1. **Initialization**: Load config, setup logging, create output directories
2. **Data Loading**: Load CSV, clean, create train/test split
3. **Cross-Validation**: 10-fold stratified split with per-fold training
4. **Evaluation**: Compute metrics (accuracy, precision, recall, F1, ROC-AUC)
5. **Best-Fold Selection**: Select highest F1-score fold
6. **Artifact Generation**:
   - CSV predictions
   - JSON metrics
   - Figures/visualizations
   - Reports/markdown analysis
7. **Optional LLM Integration** (ML pipeline only): Run LLM inference

## Shared Utilities

### Configuration (`src/utils/config.py`)
- Lazy-loading of .env variables
- Centralized access to settings
- Type-safe getters with defaults

### Logging (`src/utils/logging_config.py`)
- Consistent logging format across pipelines
- Optional file logging
- Module-level logger retrieval

### Path Management (`src/utils/path_resolver.py`)
- Standardized directory creation
- Pipeline-specific and cross-pipeline paths
- Automatic directory initialization

### Data Loading (`src/utils/data_loader.py`)
- CSV parsing with error handling
- Label normalization (handles {0,1} and {1,2})
- Train/test splitting with stratification
- Distribution analysis utilities

## Orchestration

`run_step_by_step.py` coordinates execution:

```python
STEP_ORDER = [
    "machine-learning",
    "deep-learning", 
    "transformer",
    "llm-groq",           # Optional
    "cross-comparison"    # Aggregates all results
]
```

Each step:
1. Logs execution info
2. Runs as subprocess (isolation)
3. Captures exit code and timing
4. Continues or stops based on --continue-on-error flag

## Results Structure

```
results/
├── machine_learning/
│   ├── artifacts/       # JSON metrics, CSV predictions
│   ├── figures/         # PNG/PDF visualizations
│   ├── models/          # Pickled model files
│   └── reports/         # Markdown analysis
├── deep_learning/       # [Same structure]
├── transformer/         # [Same structure]
├── llm/                 # LLM-specific outputs
│   ├── groq_llm_metrics.json
│   └── groq_*_predictions.csv
├── artifacts/           # Cross-pipeline aggregates
├── figures/            # Unified comparison plots
└── reports/            # Unified comparison markdown
```

## Best-Fold Selection

Each pipeline selects best fold using F1-score as metric:

```python
best_fold = max(fold_metrics, key=lambda x: x["f1_score"])
```

This simple, interpretable approach ensures reproducibility and fair comparison across pipelines.

## Configuration Management

All settings loaded from `.env`:
- Becomes effective at module import time
- Accessed via `Config()` singleton or direct functions
- Supports environment variable override
- Provides sensible defaults

Example:
```python
from src.utils.config import Config
config = Config()
api_key = config.groq_api_key
models = config.llm_model_names
```
