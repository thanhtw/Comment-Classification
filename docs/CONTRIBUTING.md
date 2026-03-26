"""
CONTRIBUTING.md - Guidelines for developers
"""

# Contributing to Comment Classification

## Project Structure

```
project/
├── src/
│   ├── __init__.py                    # Package initialization
│   ├── models/                        # All pipeline implementations
│   │   ├── __init__.py
│   │   ├── machine_learning_pipeline.py
│   │   ├── deep_learning_pipeline.py
│   │   ├── transformer_pipeline.py
│   │   ├── llm_groq_inference.py
│   │   └── cross_pipeline_best_fold_report.py
│   └── utils/                        # Shared utilities
│       ├── __init__.py
│       ├── config.py                 # Configuration management
│       ├── logging_config.py         # Logging setup
│       ├── path_resolver.py          # Path management
│       └── data_loader.py            # Data loading utilities
├── docs/                            # Documentation
├── data/                            # Input datasets
├── results/                         # Generated outputs
├── run_step_by_step.py              # Pipeline orchestration
├── .env.example                     # Configuration template
└── README.md                        # Project overview
```

## Adding a New Pipeline

1. Create a new file in `src/models/pipeline_name.py`
2. Use utilities from `src/utils/`:
   - `config.py` for environment setup
   - `path_resolver.py` for directory management
   - `logging_config.py` for logging
   - `data_loader.py` for data loading

Example:

```python
from src.utils.config import Config
from src.utils.logging_config import setup_logging
from src.utils.path_resolver import get_pipeline_results_dirs

logger = setup_logging(__name__)
config = Config()
results_dirs = get_pipeline_results_dirs("my_pipeline")
```

## Running Pipelines

```bash
# All pipelines
python run_step_by_step.py

# Specific pipelines
python run_step_by_step.py --only machine-learning transformer

# Continue on error
python run_step_by_step.py --continue-on-error
```

## Code Style

- Use type hints for function arguments and returns
- Add docstrings to all functions and classes
- Follow PEP 8 naming conventions
- Keep functions focused and testable

## Configuration

All configuration is managed via `.env`:
- Copy `.env.example` to `.env`
- Each variable is documented with its purpose
- Configuration is lazily loaded on first access
