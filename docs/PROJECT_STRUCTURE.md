"""
PROJECT_STRUCTURE.md - Visual guide to directory layout
"""

# Project Structure

```
Comment-Classification/
│
├── README.md                          # Main project overview
├── run_step_by_step.py               # Pipeline orchestration script
├── .env.example                      # Configuration template
├── .env                              # (not committed) Runtime configuration
├── .gitignore                        # Git ignore patterns
│
├── src/                              # Source code package
│   ├── __init__.py                   # Package initialization
│   │
│   ├── models/                       # Model pipepelines
│   │   ├── __init__.py
│   │   ├── machine_learning_pipeline.py      # SVM, NB, RF
│   │   ├── deep_learning_pipeline.py         # LSTM, BiLSTM
│   │   ├── transformer_pipeline.py           # BERT, RoBERTa
│   │   ├── llm_groq_inference.py             # Groq LLM inference
│   │   └── cross_pipeline_best_fold_report.py # Unified comparison
│   │
│   └── utils/                        # Shared utilities
│       ├── __init__.py
│       ├── config.py                 # Configuration management
│       ├── logging_config.py         # Logging setup
│       ├── path_resolver.py          # Directory management
│       └── data_loader.py            # Data loading utilities
│
├── docs/                             # Documentation
│   ├── SETUP.md                      # Installation guide
│   ├── ARCHITECTURE.md               # System design
│   ├── CONTRIBUTING.md               # Developer guidelines
│   └── PROJECT_STRUCTURE.md          # This file
│
├── data/                             # Input datasets
│   └── Dataset.csv                   # (user-provided) Training data
│
└── results/                          # Generated outputs (git-ignored)
    ├── machine_learning/             # ML pipeline results
    │   ├── artifacts/                # JSON metrics, CSV predictions
    │   ├── figures/                  # PNG/PDF visualizations
    │   ├── models/                   # Trained model pickles
    │   └── reports/                  # Markdown analysis
    │
    ├── deep_learning/                # DL pipeline results (same structure)
    │
    ├── transformer/                  # Transformer results (same structure)
    │
    ├── llm/                          # LLM-specific outputs
    │   ├── groq_llm_metrics.json
    │   └── groq_*_predictions.csv
    │
    ├── artifacts/                    # Cross-pipeline aggregates
    │   └── cross_pipeline_best_fold_metrics.csv
    │
    ├── figures/                      # Unified comparison plots
    │   └── cross_pipeline_best_fold_f1_comparison.png
    │
    └── reports/                      # Unified reports
        └── cross_pipeline_best_fold_report.md
```

## Key Design Decisions

### 1. Consolidated Source Code (`src/`)
- **models/**: All pipeline implementations in one place
- **utils/**: Shared utilities reduce duplication
- Benefits: Easy to navigate, simple imports, clear dependencies

### 2. Separated Results (`results/`)
- Pipeline-specific subdirectories with standard structure
- All outputs outside source tree (clean git history)
- Cross-pipeline data in root of results/
- Benefits: Clear organization, easy cleanup

### 3. Centralized Configuration (`src/utils/config.py`)
- Single source of truth for settings
- Lazy loading on first access
- Type-safe getters with sensible defaults
- Benefits: No duplication, environment-based customization

### 4. Standardized Directory Resolution
- Pipeline-specific dirs via `get_pipeline_results_dirs()`
- LLM-specific dir via `get_llm_results_dir()`
- Cross-pipeline dirs via `get_cross_pipeline_dirs()`
- Benefits: DRY principle, automatic dir creation

### 5. Documentation in `docs/`
- SETUP.md: Installation and environment
- ARCHITECTURE.md: System design and data flow
- CONTRIBUTING.md: Developer guidelines
- Benefits: New developers can onboard quickly

## File Size Guide

Typical file sizes:
- Each pipeline script: 2-5 KB
- Utility modules: 2-3 KB each
- Results per pipeline: 5-50 MB (avg 20 MB)
- Total project with results: ~100-300 MB

## Access Patterns

### Configuration
```python
from src.utils.config import Config, get_llm_model_names
config = Config()  # Singleton
api_key = config.groq_api_key
```

### Paths
```python
from src.utils.path_resolver import get_pipeline_results_dirs
dirs = get_pipeline_results_dirs("machine_learning")
artifacts_path = dirs["artifacts"]
```

### Logging
```python
from src.utils.logging_config import setup_logging
logger = setup_logging(__name__)
logger.info("Pipeline started")
```

### Data
```python
from src.utils.data_loader import load_and_clean_data, create_train_test_split
data = load_and_clean_data()
texts_train, texts_test, labels_train, labels_test = create_train_test_split(data)
```

## Adding New Pipelines

1. Create `src/models/my_pipeline.py`
2. Use utilities from `src/utils/` for configuration, logging, paths
3. Update `run_step_by_step.py` to register new pipeline
4. Results automatically placed in `results/my_pipeline/`

Example:
```python
# src/models/my_pipeline.py
from src.utils.config import Config
from src.utils.logging_config import setup_logging
from src.utils.path_resolver import get_pipeline_results_dirs

logger = setup_logging(__name__)
config = Config()
dirs = get_pipeline_results_dirs("my_pipeline")
```

## Maintenance

### Cleaning Results
```bash
rm -rf results/
# Re-run pipelines to regenerate
```

### Updating Dependencies
```bash
pip freeze > requirements.txt
git add requirements.txt
```

### Adding Documentation
- Add .md files to `docs/`
- Update this file if structure changes
- Keep examples in documentation current
