# Comment-Classification

This repository is organized by pipeline architecture.

## Project Structure

- src/models/: all model pipeline code
  - machine_learning_pipeline.py
  - deep_learning_pipeline.py
  - transformer_pipeline.py
- data/: input datasets
- results/: generated outputs only
- run_step_by_step.py: pipeline runner
- .env: runtime config

## Configuration

Environment settings are stored in .env:

- PROJECT_ROOT: project root path (default .)
- DATA_FILE: dataset path used by pipelines
- RESULTS_ROOT: results root folder
- ML_USE_GPU: machine-learning pipeline toggle

## Run Pipelines

Run all pipelines:

```bash
python run_step_by_step.py
```

Run selected pipelines:

```bash
python run_step_by_step.py --only machine-learning
python run_step_by_step.py --only deep-learning transformer
```

Continue on failure:

```bash
python run_step_by_step.py --continue-on-error
```
