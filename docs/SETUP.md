"""
SETUP.md - Installation and environment setup guide
"""

# Setup and Installation

## Prerequisites

- Python 3.10+
- pip or conda
- ~5GB disk space for results

## Installation

### Step 1: Clone and Setup

```bash
git clone <repository>
cd Comment-Classification
```

### Step 2: Create Virtual Environment

**Using venv:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

**Using conda:**
```bash
conda create -n comment-classification python=3.10
conda activate comment-classification
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
# Core dependencies
pip install pandas numpy scikit-learn imbalanced-learn

# Visualization
pip install matplotlib seaborn

# NLP (Chinese segmentation)
pip install jieba

# Deep Learning
pip install torch transformers datasets evaluate

# LLM support (optional)
pip install groq
```

### Step 4: Configure Environment

```bash
# Copy example configuration
cp .env.example .env

# Edit .env with your settings
nano .env  # or your preferred editor
```

Key variables:
- `PROJECT_ROOT`: Project directory (standard: `.`)
- `DATA_FILE`: Path to dataset CSV
- `Groq_API_KEY`: Required only if running LLM pipeline
- `LLM_MODEL_NAMES`: Groq models to use (comma-separated)

### Step 5: Verify Setup

```bash
python -c "import pandas, numpy, torch; print('✓ Core dependencies OK')"
python -c "from src.utils.config import Config; c = Config(); print('✓ Config loading OK')"
```

## Running Pipelines

### First Run

```bash
# Run all pipelines (takes ~1-2 hours depending on hardware)
python run_step_by_step.py
```

### Testing Specific Pipeline

```bash
# Machine Learning only (fastest, ~5-10 minutes)
python run_step_by_step.py --only machine-learning

# With error continuation
python run_step_by_step.py --only machine-learning --continue-on-error
```

## Data Preparation

### Expected Format

CSV file with columns:
```
text, label
"评论文本1", 0
"评论文本2", 1
...
```

- `text`: Chinese comment (required)
- `label`: Binary (0 or 1, or 1 and 2 - will be normalized)

### Example Data

Sample data is located at `data/Dataset.csv`. Replace with your own data or use as reference.

## Troubleshooting

### NumPy/PyArrow Compatibility

If you encounter:
```
AttributeError: _ARRAY_API not found
```

Solution: Use conda with compatible NumPy 1.x:
```bash
conda create -n comment-classification python=3.10
conda install numpy=1.26 pandas scikit-learn
```

### CUDA/GPU Issues

ML pipeline runs on CPU by default. To force CPU:
```bash
export ML_USE_GPU=0
python run_step_by_step.py
```

### Out of Memory

Reduce batch sizes in pipeline configs or limit data sample:
```python
# In pipeline file
max_samples = 500  # Process first 500 samples
```

### LLM Pipeline Fails

Ensure:
1. `Groq_API_KEY` is set in `.env`
2. Models in `LLM_MODEL_NAMES` are available on your Groq account
3. LLM pipeline is optional; other pipelines continue if it fails

## Verify Installation

```bash
# Check all imports
python -c "
from src.utils.config import Config
from src.utils.logging_config import setup_logging
from src.utils.path_resolver import get_pipeline_results_dirs
from src.utils.data_loader import load_and_clean_data
print('✓ All utilities OK')
"

# Test configuration loading
python -c "
from src.utils.config import Config
c = Config()
print(f'Project root: {c.project_root}')
print(f'Data file: {c.data_file}')
print(f'Results root: {c.results_root}')
print('✓ Configuration OK')
"

# Test runner
python run_step_by_step.py --help
```

## Performance Tips

- **First run is slow**: Transformers download (~2GB) on first use
- **Enable GPU** if available (modify ML_USE_GPU in .env or pipeline file)
- **Limit samples** for testing: modify LLM_MAX_SAMPLES in .env
- **Parallel runs**: Pipelines can run independently in separate processes
