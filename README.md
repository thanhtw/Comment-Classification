# Comment Classification Pipelines

Binary Chinese comment classification using classical ML, deep learning, transformers, and LLM methods with 10-fold cross-validation.

## Quick Start

```bash
# Setup
cp .env.example .env
# Edit .env with your configuration

# Run all pipelines
python run_step_by_step.py

# Run specific pipelines
python run_step_by_step.py --only machine-learning deep-learning

# Continue after errors
python run_step_by_step.py --continue-on-error
```

## Pipelines

| Pipeline | Location | Models | Features |
|----------|----------|--------|----------|
| **Machine Learning** | `src/models/machine_learning_pipeline.py` | SVM, Naive Bayes, Random Forest | 10-fold CV, SMOTE, class weights |
| **Deep Learning** | `src/models/deep_learning_pipeline.py` | LSTM, BiLSTM | 10-fold CV, attention, balancing |
| **Transformer** | `src/models/transformer_pipeline.py` | BERT, RoBERTa | 10-fold CV, weighted loss, HF Trainer |
| **LLM** (optional) | `src/models/llm_groq_inference.py` | Groq models | Zero-shot + few-shot, configurable |
| **Cross-Pipeline Report** | `src/models/cross_pipeline_best_fold_report.py` | — | Unified comparison & metrics |

## Output Structure

```
results/
├── machine_learning/         # ML pipeline
├── deep_learning/           # Deep learning pipeline  
├── transformer/             # Transformer pipeline
├── llm/                     # LLM results (optional)
├── artifacts/               # Cross-pipeline metrics
├── figures/                 # Comparison plots
└── reports/                 # Markdown reports

# Each pipeline subdirectory contains:
├── artifacts/               # Metrics, predictions (JSON/CSV)
├── figures/                 # Visualizations (PNG/PDF)
├── models/                  # Trained model files
└── reports/                 # Analysis reports (MD)
```

## Configuration

Edit `.env` (copy from `.env.example`):

- `PROJECT_ROOT`: Project directory (default: `.`)
- `DATA_FILE`: Dataset CSV path (default: `./data/Dataset.csv`)
- `RESULTS_ROOT`: Output directory (default: `./results`)
- `Groq_API_KEY`: Groq API key (required for LLM pipeline only)
- `LLM_MODEL_NAMES`: Comma-separated model IDs for LLM
- `LLM_MAX_SAMPLES`: Max samples for LLM evaluation (default: 200)
- `ML_USE_GPU`: Enable GPU for sklearn (0=CPU, 1=GPU)
- `PYTHONUTF8`: Unicode support (default: 1)

## Dependencies

```bash
# Core
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn jieba

# Deep learning  
pip install torch transformers datasets evaluate

# LLM (optional)
pip install groq
```

## Data Format

CSV format with columns:
- `text`: Chinese comment text
- `label`: Binary label (0 or 1)

## Notes

- LLM pipeline is optional; other pipelines run independently
- All results saved to `results/` (not source code directories)
- Best-fold selected per model via F1 score
- Cross-pipeline report compares all best-fold metrics
