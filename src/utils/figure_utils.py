"""Professional figure utilities for publication-quality visualizations.

This module provides:
- Consistent styling across all pipelines
- Multi-format figure export (PNG, PDF)
- Common plot templates (fold comparison, metrics, confusion matrix)
- High DPI output (600 DPI) for print/publication
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Professional color palettes
PALETTE_DEEP = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # Blue, Orange, Green
PALETTE_FOLD_COMPARISON = ["#3498db", "#e74c3c"]  # Blue vs Red for model comparison
COLORS_MODELS = {
    "SVM": "#1f77b4",           # Blue
    "NB": "#ff7f0e",            # Orange
    "RF": "#2ca02c",            # Green
    "LSTM": "#d62728",          # Red
    "BiLSTM": "#9467bd",        # Purple
    "BERT": "#8c564b",          # Brown
    "RoBERTa": "#e377c2"        # Pink
}


def setup_professional_style() -> None:
    """Apply professional publication-oriented plotting style globally."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'figure.dpi': 100,
        'savefig.dpi': 600,
        'font.size': 11,
        'font.family': 'sans-serif',
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'legend.framealpha': 0.95,
        'figure.titlesize': 14,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': ':',
        'legend.loc': 'best',
    })
    sns.set_palette("husl")


def save_figure_multi_format(
    fig: plt.Figure,
    output_path: Path,
    filename_stem: str,
    formats: Tuple[str, ...] = ("png", "pdf", "svg")
) -> None:
    """Save figure in multiple formats with consistent DPI.
    
    Args:
        fig: Matplotlib figure object
        output_path: Output directory path
        filename_stem: Filename without extension
        formats: Tuple of formats to save (png, pdf, svg)
    """
    output_path.mkdir(parents=True, exist_ok=True)
    
    for fmt in formats:
        filepath = output_path / f"{filename_stem}.{fmt}"
        fig.savefig(filepath, dpi=600, bbox_inches='tight', format=fmt)


def plot_confusion_matrix_consistent(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    output_path: Optional[Path] = None,
    filename_stem: str = "confusion_matrix",
    class_labels: Optional[List[str]] = None,
    formats: Tuple[str, ...] = ("png", "pdf"),
) -> plt.Figure:
    """Render one confusion matrix with consistent style and pixel layout.

    This function is intended to be shared by all pipelines so axis rotation,
    text placement, and sizing are identical across outputs.
    """
    setup_professional_style()

    if class_labels is None:
        class_labels = ["No-Meaningful", "Meaningful"]

    y_true_arr = np.asarray(y_true).astype(int)
    y_pred_arr = np.asarray(y_pred).astype(int)
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true_arr, y_pred_arr):
        if t in (0, 1) and p in (0, 1):
            cm[t, p] += 1

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        ax=ax,
        cbar=True,
        cbar_kws={'label': 'Count'},
        annot_kws={'fontsize': 14, 'fontweight': 'bold'},
    )

    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_xticklabels(class_labels, rotation=0, ha='center')
    ax.set_yticklabels(class_labels, rotation=0)
    ax.grid(False)

    total = int(np.sum(cm))
    accuracy = float(np.trace(cm) / total) if total > 0 else 0.0
    # Keep summary text outside axes to avoid overlap with tick labels.
    fig.text(
        0.5,
        0.03,
        f'Accuracy: {accuracy:.4f} | Samples: {total}',
        ha='center',
        va='bottom',
        fontsize=11,
        fontweight='bold',
    )

    fig.subplots_adjust(bottom=0.18, top=0.90)
    if output_path is not None:
        save_figure_multi_format(fig, output_path, filename_stem, formats)
    return fig


def plot_fold_metrics_comparison(
    fold_results_dict: Dict[str, List[Dict]],
    metric_name: str = "accuracy",
    output_path: Optional[Path] = None,
    filename_stem: str = "fold_metrics_comparison",
    models: Optional[List[str]] = None
) -> Optional[plt.Figure]:
    """Create professional fold-by-fold comparison plot.
    
    Args:
        fold_results_dict: Dict mapping model names to list of fold results
        metric_name: Metric to plot (accuracy, f1_score, precision, recall)
        output_path: Directory to save figure (None for no save)
        filename_stem: Filename without extension
        models: List of model names to include (None for all)
        
    Returns:
        Matplotlib figure object
    """
    setup_professional_style()
    
    if models is None:
        models = list(fold_results_dict.keys())
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get fold data for each model
    num_folds = len(fold_results_dict[models[0]])
    folds = np.arange(1, num_folds + 1)
    
    for model in models:
        values = [fold.get(metric_name, np.nan) for fold in fold_results_dict[model]]
        color = COLORS_MODELS.get(model, "#1f77b4")
        ax.plot(folds, values, marker='o', label=model, linewidth=2.5, 
                markersize=8, color=color, alpha=0.8)
    
    ax.set_xlabel("Fold Number", fontsize=12)
    ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f"{metric_name.replace('_', ' ').title()} Across CV Folds", fontsize=13)
    ax.set_xticks(folds)
    ax.legend(frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    if output_path:
        save_figure_multi_format(fig, output_path, filename_stem, ("png", "pdf"))
    
    return fig


def plot_metrics_panel(
    fold_results_dict: Dict[str, List[Dict]],
    output_path: Optional[Path] = None,
    filename_stem: str = "metrics_panel",
    models: Optional[List[str]] = None,
    separate: bool = False
) -> Optional[plt.Figure]:
    """Create 2x2 panel of fold metrics: accuracy, precision, recall, F1.
    
    Args:
        fold_results_dict: Dict mapping model names to list of fold results
        output_path: Directory to save figure (None for no save)
        filename_stem: Filename without extension
        models: List of model names to include (None for all)
        separate: If True, save one figure per metric instead of a 2x2 panel
        
    Returns:
        Matplotlib figure object
    """
    setup_professional_style()
    
    if models is None:
        models = list(fold_results_dict.keys())
    
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    color_cycle = sns.color_palette("tab10", n_colors=max(len(models), 3))
    model_color_map = {}
    for idx, model in enumerate(models):
        model_color_map[model] = COLORS_MODELS.get(model, color_cycle[idx % len(color_cycle)])

    if separate:
        last_fig = None
        for metric in metrics:
            fig, ax = plt.subplots(figsize=(12, 6))
            num_folds = len(fold_results_dict[models[0]])
            folds = np.arange(1, num_folds + 1)

            for model in models:
                values = [fold.get(metric, np.nan) for fold in fold_results_dict[model]]
                ax.plot(
                    folds,
                    values,
                    marker='o',
                    label=model,
                    linewidth=2.5,
                    markersize=7,
                    color=model_color_map[model],
                    alpha=0.85
                )

            metric_title = metric.replace('_', ' ').title()
            ax.set_xlabel("Fold", fontsize=11)
            ax.set_ylabel(metric_title, fontsize=11)
            ax.set_title(f"{metric_title} Across Folds", fontsize=12)
            ax.set_xticks(folds)
            ax.set_ylim([0, 1.05])
            ax.grid(True, alpha=0.3)
            ax.legend(frameon=True)
            plt.tight_layout()

            if output_path:
                save_figure_multi_format(fig, output_path, f"{filename_stem}_{metric}", ("png", "pdf"))

            last_fig = fig

        return last_fig

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    num_folds = len(fold_results_dict[models[0]])
    folds = np.arange(1, num_folds + 1)
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        for model in models:
            values = [fold.get(metric, np.nan) for fold in fold_results_dict[model]]
            color = model_color_map[model]
            ax.plot(folds, values, marker='o', label=model, linewidth=2.5,
                   markersize=7, color=color, alpha=0.8)
        
        ax.set_xlabel("Fold", fontsize=11)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
        ax.set_title(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_xticks(folds)
        ax.legend(frameon=True)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
    
    plt.suptitle("Performance Metrics Across Folds", fontsize=14, y=1.00)
    plt.tight_layout()
    
    if output_path:
        save_figure_multi_format(fig, output_path, filename_stem, ("png", "pdf"))
    
    return fig


def plot_loss_comparison(
    fold_results_dict: Dict[str, List[Dict]],
    output_path: Optional[Path] = None,
    filename_stem: str = "loss_comparison",
    models: Optional[List[str]] = None
) -> Optional[plt.Figure]:
    """Create fold-by-fold loss/log_loss comparison plot.
    
    Args:
        fold_results_dict: Dict mapping model names to list of fold results
        output_path: Directory to save figure (None for no save)
        filename_stem: Filename without extension
        models: List of model names to include (None for all)
        
    Returns:
        Matplotlib figure object (None if no loss data)
    """
    setup_professional_style()
    
    if models is None:
        models = list(fold_results_dict.keys())
    
    # Check if any model has loss data
    loss_key = None
    for model in models:
        for fold in fold_results_dict[model]:
            for key in ['log_loss', 'val_loss', 'train_loss']:
                if key in fold:
                    loss_key = key
                    break
        if loss_key:
            break
    
    if not loss_key:
        return None
    
    fig, ax = plt.subplots(figsize=(12, 6))
    num_folds = len(fold_results_dict[models[0]])
    folds = np.arange(1, num_folds + 1)
    
    for model in models:
        values = [fold.get(loss_key, np.nan) for fold in fold_results_dict[model]]
        color = COLORS_MODELS.get(model, "#1f77b4")
        ax.plot(folds, values, marker='s', label=model, linewidth=2.5,
               markersize=8, color=color, alpha=0.8)
    
    ax.set_xlabel("Fold Number", fontsize=12)
    ax.set_ylabel(loss_key.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f"{loss_key.replace('_', ' ').title()} Across CV Folds", fontsize=13)
    ax.set_xticks(folds)
    ax.legend(frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        save_figure_multi_format(fig, output_path, filename_stem, ("png", "pdf"))
    
    return fig


def plot_model_comparison_bar(
    avg_metrics_dict: Dict[str, Dict[str, float]],
    metrics: List[str] = None,
    output_path: Optional[Path] = None,
    filename_stem: str = "model_comparison",
    title: str = "Model Performance Comparison"
) -> Optional[plt.Figure]:
    """Create grouped bar chart comparing models across metrics.
    
    Args:
        avg_metrics_dict: Dict mapping model names to dict of metrics
        metrics: List of metrics to compare (defaults to standard metrics)
        output_path: Directory to save figure (None for no save)
        filename_stem: Filename without extension
        title: Figure title
        
    Returns:
        Matplotlib figure object
    """
    setup_professional_style()
    
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    models = list(avg_metrics_dict.keys())
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(models))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = [avg_metrics_dict[model].get(metric, 0) for model in models]
        ax.bar(x + i * width, values, width, label=metric.replace('_', ' ').title())
    
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models)
    ax.legend(frameon=True, shadow=True)
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        save_figure_multi_format(fig, output_path, filename_stem, ("png", "pdf"))
    
    return fig


def export_fold_metrics_csv(
    fold_results_dict: Dict[str, List[Dict]],
    output_path: Path,
    filename_stem: str = "fold_metrics"
) -> Path:
    """Export per-fold metrics to CSV for all models.
    
    Args:
        fold_results_dict: Dict mapping model names to list of fold results
        output_path: Output directory
        filename_stem: Filename without extension
        
    Returns:
        Path to saved CSV file
    """
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Collect all folds into one dataframe
    data = []
    for model_name, folds in fold_results_dict.items():
        for fold_idx, fold_metrics in enumerate(folds, 1):
            row = {'model': model_name, 'fold': fold_idx}
            row.update(fold_metrics)
            data.append(row)
    
    df = pd.DataFrame(data)
    csv_path = output_path / f"{filename_stem}.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8')
    
    return csv_path



