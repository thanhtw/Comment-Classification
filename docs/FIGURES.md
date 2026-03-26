# Figure Generation and Visualization Guide

## Overview

This document describes the professional figure generation system implemented across all model pipelines. All figures are generated with publication-quality standards (600 DPI, multi-format export, consistent styling).

## Professional Styling Features

### Global Style Configuration
- **Framework**: Seaborn + Matplotlib
- **DPI**: 600 for all saved figures (PNG, PDF, SVG)
- **Color Palette**: Consistent, colorblind-friendly colors:
  - **SVM**: `#1f77b4` (Blue)
  - **Naive Bayes**: `#ff7f0e` (Orange)
  - **Random Forest**: `#2ca02c` (Green)
  - **LSTM**: `#d62728` (Red)
  - **BiLSTM**: `#9467bd` (Purple)
  - **BERT**: `#8c564b` (Brown)
  - **RoBERTa**: `#e377c2` (Pink)

### Figure Configuration
- Figure size: Automatically scaled (12-16 inches)
- Font sizes: 10-14pt (readable in print)
- Grid lines: Enabled with alpha=0.3 (subtle)
- Legend: Always displayed with frame and shadow
- Margins: "tight" bbox to minimize whitespace

## Figure Export Formats

All figures are automatically saved in **three formats**:
1. **PNG**: 600 DPI, lossless, ideal for presentations and web
2. **PDF**: Vector format, ideal for publications and printing
3. **SVG**: Scalable vector, ideal for further editing

Location: `results/{pipeline}/figures/`

## Unified Figure Utilities

Location: `src/utils/figure_utils.py`

### Available Functions

#### 1. `plot_metrics_panel()`
Creates a 2x2 panel showing:
- Accuracy across folds
- Precision across folds
- Recall across folds
- F1-Score across folds

**Usage:**
```python
from src.utils.figure_utils import plot_metrics_panel
fig = plot_metrics_panel(fold_results_dict, output_path, "metrics_panel")
```

**Output:** `metrics_panel.{png,pdf,svg}`

---

#### 2. `plot_fold_metrics_comparison()`
Creates line plot tracking a single metric across all folds for all models.

**Usage:**
```python
fig = plot_fold_metrics_comparison(
    fold_results_dict,
    metric_name="accuracy",
    output_path=output_path,
    filename_stem="fold_accuracy"
)
```

**Output:** `fold_accuracy.{png,pdf,svg}`

---

#### 3. `plot_loss_comparison()`
Creates line plot comparing training/validation loss across folds.

**Supported loss keys:**
- `log_loss` (ML models)
- `val_loss` (Deep Learning)
- `train_loss` (alternative)

**Usage:**
```python
fig = plot_loss_comparison(
    fold_results_dict,
    output_path=output_path,
    filename_stem="fold_loss"
)
```

**Output:** `fold_loss.{png,pdf,svg}` (or None if no loss data)

---

#### 4. `plot_model_comparison_bar()`
Creates grouped bar chart comparing average metrics across models.

**Usage:**
```python
fig = plot_model_comparison_bar(
    avg_metrics_dict,
    metrics=['accuracy', 'precision', 'recall', 'f1_score'],
    output_path=output_path,
    filename_stem="model_comparison",
    title="Model Performance Comparison"
)
```

**Output:** `model_comparison.{png,pdf,svg}`

---

#### 5. `export_fold_metrics_csv()`
Exports all per-fold metrics to CSV for reproducibility and external analysis.

**Usage:**
```python
csv_path = export_fold_metrics_csv(
    fold_results_dict,
    output_path=output_path,
    filename_stem="fold_metrics"
)
```

**Output:** `fold_metrics.csv`

**CSV Structure:**
```
model,fold,accuracy,precision,recall,f1_score,log_loss,...
SVM,1,0.8234,0.7956,0.8012,0.7983,0.3456
SVM,2,0.8145,0.7823,0.7934,0.7878,0.3567
...
```

---

#### 6. `plot_confusion_matrix_panel()`
Creates confusion matrix visualizations for multiple models.

**Usage:**
```python
cm_dict = {
    'SVM': confusion_matrix(...),
    'NB': confusion_matrix(...)
}
fig = plot_confusion_matrix_panel(
    cm_dict,
    output_path=output_path,
    filename_stem="confusion_matrices"
)
```

**Output:** `confusion_matrices.{png,pdf,svg}`

---

#### 7. `plot_class_distribution()`
Creates side-by-side class distribution comparison.

**Usage:**
```python
y_data = {
    'Train': y_train,
    'Test': y_test,
    'CV': y_cv
}
fig = plot_class_distribution(
    y_data,
    output_path=output_path,
    filename_stem="class_distribution"
)
```

**Output:** `class_distribution.{png,pdf,svg}`

---

#### 8. `save_figure_multi_format()`
Low-level function to save a figure in multiple formats.

**Usage:**
```python
save_figure_multi_format(
    fig,
    output_path=Path("results/figures"),
    filename_stem="my_plot",
    formats=("png", "pdf", "svg")
)
```

---

#### 9. `setup_professional_style()`
Apply professional styling globally to all subsequent plots.

**Usage:**
```python
from src.utils.figure_utils import setup_professional_style
setup_professional_style()
# All plots created after this call will use professional styling
```

## Per-Pipeline Figure Generation

### Machine Learning Pipeline

**Files Generated:**
- `ml_fold_metrics.csv` - Per-fold metrics for all models
- `ml_metrics_panel.{png,pdf,svg}` - 2x2 metrics across folds
- `ml_model_comparison.{png,pdf,svg}` - Average metrics comparison
- `comprehensive_model_comparison_balanced.png` - Legacy combined figure
- `smote_before_after_distribution_final_train.png` - Class distribution

**Triggering Function:**
```python
create_publication_ready_plots(
    avg_metrics_dict,
    fold_results_dict,
    output_dir=str(result_dirs['figures'])
)
```

---

### Deep Learning Pipeline (LSTM/BiLSTM)

**Files Generated:**
- `dl_fold_metrics.csv` - Per-fold metrics for all models
- `dl_metrics_panel.{png,pdf,svg}` - 2x2 metrics across folds
- `dl_loss_comparison.{png,pdf,svg}` - Loss comparison across folds
- `dl_model_comparison.{png,pdf,svg}` - Average metrics comparison
- `LSTM_BiLSTM_Comparison_Report.md` - Markdown report with figure references

**Triggering Function:**
```python
create_comparison_plots(
    bilstm_results,
    lstm_results,
    bilstm_avg_metrics,
    lstm_avg_metrics,
    output_path=str(output_dir / "lstm_bilstm_comparison.png")
)
```

---

### Transformer Pipeline (BERT/RoBERTa)

**Files Generated:**
- `transformer_fold_metrics.csv` - Per-fold metrics for all models
- `transformer_metrics_panel.{png,pdf,svg}` - 2x2 metrics across folds
- `transformer_loss_comparison.{png,pdf,svg}` - Validation loss comparison
- `figure_transformer_best_fold_comparison_panel.{png,pdf,svg}` - Best-fold metrics
- `figure_transformer_best_fold_confusion_matrices.{png,pdf,svg}` - Confusion matrices

**Triggering Function:**
```python
pipeline._plot_model_comparison(comparison_df)  # During pipeline.run()
```

---

### Cross-Pipeline Comparison

**Files Generated:**
- `cross_pipeline_best_fold_f1_comparison.{png,pdf}` - F1-score comparison across all pipelines

**Location:** `results/figures/`

---

## Per-Fold Metrics Export

All pipelines now export **per-fold metrics as CSV** for reproducibility and external analysis.

### CSV Format

Each pipeline generates a CSV with columns:
- `model`: Model name (e.g., "SVM", "LSTM", "BERT")
- `fold`: Fold number (1-10)
- `accuracy`: Accuracy score on test fold
- `precision`: Weighted precision
- `recall`: Weighted recall
- `f1_score`: Weighted F1 score
- Additional metrics: `roc_auc`, `inference_time`, etc. (varies by pipeline)

### Access CSV Files

**Machine Learning:**
```
results/machine_learning/figures/ml_fold_metrics.csv
```

**Deep Learning:**
```
results/deep_learning/figures/dl_fold_metrics.csv
```

**Transformer:**
```
results/transformer/figures/transformer_fold_metrics.csv
```

### Using CSV Data

```python
import pandas as pd

# Load ML fold metrics
ml_metrics = pd.read_csv("results/machine_learning/figures/ml_fold_metrics.csv")

# Get best fold for each model
best_per_model = ml_metrics.loc[ml_metrics.groupby('model')['f1_score'].idxmax()]

# Analyze fold stability
for model in ml_metrics['model'].unique():
    model_data = ml_metrics[ml_metrics['model'] == model]
    std_dev = model_data['accuracy'].std()
    print(f"{model}: accuracy std_dev = {std_dev:.4f}")
```

---

## Figure Examples and Descriptions

### Example 1: 2x2 Metrics Panel

Shows accuracy, precision, recall, F1 across all 10 folds.

```
┌─────────────────┬─────────────────┐
│   Accuracy      │   Precision     │
│  SVM vs LSTM    │  SVM vs LSTM    │
│  (fold by fold) │  (fold by fold) │
├─────────────────┼─────────────────┤
│   Recall        │   F1-Score      │
│  SVM vs LSTM    │  SVM vs LSTM    │
│  (fold by fold) │  (fold by fold) │
└─────────────────┴─────────────────┘
```

### Example 2: Loss Comparison

Shows training/validation loss trends across folds.

```
Loss Across Folds
    │
  │ │ BiLSTM (purple)
  │ │ LSTM (red)
  │ │
  │ └─────────────────→ Fold
```

### Example 3: Model Comparison Bar Chart

Groups metrics by model for average performance.

```
Accuracy Comparison
    │
1.0 │  ▄▄  ▄▄  ▄▄
    │  ██  ██  ██
    │  ██  ██  ██
    └─────────────→
      SVM NB  RF
```

## Integration with Markdown Reports

All pipelines can reference figures in their markdown reports:

```markdown
## Figures

### Per-Fold Performance
![Metrics Panel](./figures/ml_metrics_panel.png)

### Model Comparison
![Model Comparison](./figures/ml_model_comparison.pdf)

### Detailed Metrics
See detailed per-fold metrics in [ml_fold_metrics.csv](./figures/ml_fold_metrics.csv)
```

---

## Best Practices

### When Creating Custom Figures

1. **Always use UTC professional style:**
   ```python
   from src.utils.figure_utils import setup_professional_style
   setup_professional_style()
   ```

2. **Save in multiple formats:**
   ```python
   from src.utils.figure_utils import save_figure_multi_format
   save_figure_multi_format(fig, output_path, "my_figure")
   ```

3. **Use standardized colors:**
   ```python
   from src.utils.figure_utils import COLORS_MODELS
   ax.plot(data, color=COLORS_MODELS['SVM'])
   ```

4. **Always close figures after saving:**
   ```python
   plt.savefig(path)
   plt.close(fig)  # Prevents memory leaks
   ```

### For Publication

1. Export as **PDF** for vector quality
2. Use **SVG** for web and further editing
3. Include figure **captions** in markdown
4. Reference the **CSV data** in supplementary materials

---

## Troubleshooting

### Issue: Figures not appearing or appear blank

**Solution:**
```bash
python -c "import matplotlib; matplotlib.use('Agg')"  # Set non-interactive backend
```

Add to top of pipeline scripts:
```python
import matplotlib
matplotlib.use('Agg')  # Use non-display backend
```

### Issue: DPI too low or text too small

**Solution:** Update DPI in figure_utils.py or use:
```python
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 600
```

### Issue: Colors not loading consistently

**Solution:** Ensure `setup_professional_style()` is called before plotting:
```python
from src.utils.figure_utils import setup_professional_style
setup_professional_style()  # Must be called first
```

### Issue: Memory leak with many figures

**Solution:** Always close figures:
```python
plt.close(fig)      # Close specific figure
plt.close('all')    # Close all figures
```

---

## Future Improvements

- [ ] Add interactive figures (Plotly/Altair)
- [ ] Generate auto-embedded figures in notebooks
- [ ] Add 3D visualization options
- [ ] Create figure comparison tool
- [ ] Add statistical significance annotations
- [ ] Generate animated fold transitions (GIF)

---

## References

- Matplotlib documentation: https://matplotlib.org/
- Seaborn documentation: https://seaborn.pydata.org/
- Publication figure standards: https://journals.plos.org/plosone/s/figures
