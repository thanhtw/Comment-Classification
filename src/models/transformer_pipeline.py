import json
import logging
import os
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
)

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def log_info(message: str) -> None:
    """Project-wide lightweight logger wrapper."""
    logger.info(message)


def get_project_root() -> Path:
    """Resolve project root from env or repository layout."""
    env_root = os.getenv("PROJECT_ROOT", "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def get_results_dirs() -> dict:
    """Create and return standardized output directories for transformer pipeline."""
    project_root = get_project_root()
    base = project_root / "results" / "transformer"
    dirs = {
        "base": base,
        "figures": base / "figures",
        "artifacts": base / "artifacts",
        "models": base / "models",
        "reports": base / "reports",
        "runs": base / "runs",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


class WeightedTrainer(Trainer):
    """Hugging Face trainer with class-weighted cross-entropy loss."""

    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.class_weights is not None:
            weight_tensor = torch.tensor(
                list(self.class_weights.values()),
                dtype=torch.float32,
                device=logits.device,
            )
            loss_fct = torch.nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()

        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


class TransformerComparisonPipeline:
    """Class-based pipeline for BERT vs RoBERTa Chinese text classification."""

    def __init__(self) -> None:
        self.result_dirs = get_results_dirs()
        project_root = get_project_root()
        self.data_file = Path(os.getenv("DATA_FILE", str(project_root / "data" / "Dataset.csv")))

        self.models_config = {
            "BERT-Chinese": {
                "model_name": "google-bert/bert-base-chinese",
                "model_class": BertForSequenceClassification,
                "tokenizer_class": BertTokenizer,
            },
            "RoBERTa-Chinese": {
                "model_name": "hfl/chinese-roberta-wwm-ext",
                "model_class": AutoModelForSequenceClassification,
                "tokenizer_class": AutoTokenizer,
            },
        }

        self.training_config = {
            "num_train_epochs": 3,
            "per_device_train_batch_size": 8,
            "per_device_eval_batch_size": 8,
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "fp16": True,
            "logging_steps": 10,
            "evaluation_strategy": "epoch",
            "save_strategy": "epoch",
            "load_best_model_at_end": True,
            "metric_for_best_model": "f1",
            "save_total_limit": 2,
            "seed": 42,
        }

        self.data = None
        self.train_data = None
        self.test_data = None
        self.train_dataset = None
        self.test_dataset = None
        self.class_weights = None
        self.results = {}
        self.training_artifacts = {}
        self.figure_dpi = 600
        self.figure_formats = ("png", "pdf")

    def _set_publication_style(self) -> None:
        """Apply publication-ready figure style suitable for journal submissions."""
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams.update(
            {
                "font.family": "DejaVu Serif",
                "font.size": 10,
                "axes.titlesize": 12,
                "axes.labelsize": 10,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "legend.fontsize": 9,
                "figure.titlesize": 13,
                "axes.spines.top": False,
                "axes.spines.right": False,
            }
        )

    def _save_figure_multi_format(self, fig: plt.Figure, filename_stem: str) -> None:
        """Save one figure in all configured publication formats."""
        for fmt in self.figure_formats:
            fig.savefig(
                self.result_dirs["figures"] / f"{filename_stem}.{fmt}",
                dpi=self.figure_dpi,
                bbox_inches="tight",
            )

    @staticmethod
    def compute_metrics(eval_pred):
        """Compute weighted metrics for trainer evaluation."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1).astype(int)
        labels = labels.astype(int)

        return {
            "accuracy": float(np.mean(predictions == labels)),
            "precision": float(precision_score(labels, predictions, average="weighted", zero_division=0)),
            "recall": float(recall_score(labels, predictions, average="weighted", zero_division=0)),
            "f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
        }

    @staticmethod
    def _calculate_class_weights(labels: np.ndarray) -> dict:
        """Calculate class weights for imbalanced data."""
        class_weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
        return {i: weight for i, weight in enumerate(class_weights)}

    @staticmethod
    def _build_dataset_info(data: pd.DataFrame, train_data: pd.DataFrame, test_data: pd.DataFrame, class_weights: dict) -> dict:
        """Create dataset summary metadata for reporting."""
        return {
            "total_samples": len(data),
            "train_size": len(train_data),
            "test_size": len(test_data),
            "class_distribution": dict(data["label"].value_counts().sort_index()),
            "class_weights": class_weights,
        }

    def _load_and_prepare_data(self) -> None:
        """Load CSV, encode labels, split train/test, and build HF datasets."""
        self.data = pd.read_csv(self.data_file, encoding="utf-8")
        if "text" not in self.data.columns or "label" not in self.data.columns:
            raise ValueError("Dataset must contain 'text' and 'label' columns.")

        log_info(f"Dataset loaded: {self.data.shape}")
        log_info(f"Label distribution:\n{self.data['label'].value_counts()}")

        # Deterministic label mapping for reproducibility.
        label_to_int = {label: idx for idx, label in enumerate(sorted(self.data["label"].unique()))}
        self.data["label"] = self.data["label"].map(label_to_int)

        self.train_data, self.test_data = train_test_split(
            self.data,
            test_size=0.2,
            random_state=42,
            shuffle=True,
        )

        log_info(f"Training set size: {len(self.train_data)}")
        self.class_weights = self._calculate_class_weights(self.train_data["label"].values)
        log_info(f"Class weights: {self.class_weights}")

        self.train_data.to_csv(self.result_dirs["artifacts"] / "train_set.csv", index=False)
        self.test_data.to_csv(self.result_dirs["artifacts"] / "test_set.csv", index=False)

        self.train_dataset = Dataset.from_pandas(self.train_data[["text", "label"]])
        self.test_dataset = Dataset.from_pandas(self.test_data[["text", "label"]])
        log_info(f"Training set: {len(self.train_dataset)} samples")
        log_info(f"Test set: {len(self.test_dataset)} samples")

    def _train_one_model(self, model_name: str, config: dict) -> None:
        """Train, evaluate, save model artifacts, and capture per-model outputs."""
        log_info(f"\n{'=' * 50}")
        log_info(f"Training {model_name}")
        log_info(f"{'=' * 50}")

        tokenizer = config["tokenizer_class"].from_pretrained(config["model_name"])
        model = config["model_class"].from_pretrained(
            config["model_name"],
            num_labels=2,
            hidden_dropout_prob=0.2,
            attention_probs_dropout_prob=0.1,
        )

        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=200)

        tokenized_train = self.train_dataset.map(tokenize_function, batched=True)
        tokenized_test = self.test_dataset.map(tokenize_function, batched=True)

        training_args = TrainingArguments(
            output_dir=str(self.result_dirs["runs"] / model_name.replace("-", "_")),
            **self.training_config,
        )

        trainer = WeightedTrainer(
            class_weights=self.class_weights,
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            compute_metrics=self.compute_metrics,
        )

        log_info(f"Starting training for {model_name}...")
        train_result = trainer.train()
        eval_result = trainer.evaluate()

        predictions = trainer.predict(tokenized_test)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = predictions.label_ids

        self.results[model_name] = {
            "eval_metrics": eval_result,
            "predictions": y_pred,
            "true_labels": y_true,
        }

        self.training_artifacts[model_name] = {
            "train_metrics": train_result.metrics,
            "eval_metrics": eval_result,
            "log_history": trainer.state.log_history,
            "model_dir": str(self.result_dirs["models"] / model_name.replace("-", "_")),
        }

        prediction_df = pd.DataFrame(
            {
                "sample_index": np.arange(len(y_true)),
                "true_label": y_true.astype(int),
                "predicted_label": y_pred.astype(int),
                "model": model_name,
            }
        )
        prediction_df.to_csv(
            self.result_dirs["artifacts"] / f"{model_name.replace('-', '_').lower()}_test_predictions.csv",
            index=False,
            encoding="utf-8",
        )

        log_info(f"\n{model_name} Results:")
        log_info(f"Accuracy: {eval_result['eval_accuracy']:.4f}")
        log_info(f"Precision: {eval_result['eval_precision']:.4f}")
        log_info(f"Recall: {eval_result['eval_recall']:.4f}")
        log_info(f"F1-Score: {eval_result['eval_f1']:.4f}")

        model_dir = self.result_dirs["models"] / model_name.replace("-", "_")
        model.save_pretrained(str(model_dir))
        tokenizer.save_pretrained(str(model_dir))

        del trainer, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _build_comparison_df(self) -> pd.DataFrame:
        """Build model comparison dataframe from evaluation metrics."""
        comparison_data = []
        for model_name, result in self.results.items():
            metrics = result["eval_metrics"]
            comparison_data.append(
                {
                    "Model": model_name,
                    "Accuracy": metrics["eval_accuracy"],
                    "Precision": metrics["eval_precision"],
                    "Recall": metrics["eval_recall"],
                    "F1-Score": metrics["eval_f1"],
                }
            )
        return pd.DataFrame(comparison_data)

    def _plot_model_comparison(self, comparison_df: pd.DataFrame) -> None:
        """Create publication-ready combined and per-metric comparison charts."""
        self._set_publication_style()

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
        colors = ["#1f4e79", "#b35c1e"]

        for i, metric in enumerate(metrics):
            ax = axes[i // 2, i % 2]
            bars = ax.bar(comparison_df["Model"], comparison_df[metric], color=colors, width=0.6)
            ax.set_title(f"{metric} Comparison")
            ax.set_ylabel(metric)
            ax.set_ylim(0, 1)
            ax.grid(True, axis="y", alpha=0.25)
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2.0, height + 0.01, f"{height:.4f}", ha="center", va="bottom")

        plt.tight_layout()
        self._save_figure_multi_format(fig, "figure_transformer_model_comparison_panel")
        plt.close()

        # Export each metric as an individual standalone figure for manuscript layout flexibility.
        for metric in metrics:
            fig_metric, ax_metric = plt.subplots(figsize=(6.5, 4.5))
            bars = ax_metric.bar(comparison_df["Model"], comparison_df[metric], color=colors, width=0.58)
            ax_metric.set_title(f"{metric} Comparison")
            ax_metric.set_ylabel(metric)
            ax_metric.set_ylim(0, 1)
            ax_metric.grid(True, axis="y", alpha=0.25)
            for bar in bars:
                height = bar.get_height()
                ax_metric.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01,
                    f"{height:.4f}",
                    ha="center",
                    va="bottom",
                )
            plt.tight_layout()
            stem = f"figure_transformer_{metric.lower().replace('-', '_').replace(' ', '_')}"
            self._save_figure_multi_format(fig_metric, stem)
            plt.close(fig_metric)

    def _plot_confusion_matrices(self) -> None:
        """Create publication-ready combined and per-model confusion matrix plots."""
        self._set_publication_style()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for i, (model_name, result) in enumerate(self.results.items()):
            cm = confusion_matrix(result["true_labels"], result["predictions"])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Positive", "Negative"])
            disp.plot(ax=axes[i], cmap="Blues")
            axes[i].set_title(f"{model_name} Confusion Matrix")
            axes[i].grid(False)

        plt.tight_layout()
        self._save_figure_multi_format(fig, "figure_transformer_confusion_matrices_panel")
        plt.close()

        for model_name, result in self.results.items():
            fig_cm, ax_cm = plt.subplots(figsize=(5.2, 4.6))
            cm = confusion_matrix(result["true_labels"], result["predictions"])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Positive", "Negative"])
            disp.plot(ax=ax_cm, cmap="Blues", colorbar=False, values_format="d")
            ax_cm.set_title(f"{model_name} Confusion Matrix")
            ax_cm.grid(False)
            plt.tight_layout()
            stem = f"figure_transformer_confusion_matrix_{model_name.lower().replace('-', '_')}"
            self._save_figure_multi_format(fig_cm, stem)
            plt.close(fig_cm)

    def _log_classification_reports(self) -> None:
        """Log detailed classification report for each model."""
        for model_name, result in self.results.items():
            log_info(f"\n{model_name} - Detailed Classification Report:")
            log_info("-" * 50)
            log_info(
                classification_report(
                    result["true_labels"],
                    result["predictions"],
                    target_names=["Positive Feedback", "Negative/Constructive Feedback"],
                )
            )

    def generate_markdown_report(
        self,
        comparison_df: pd.DataFrame,
        dataset_info: dict,
        report_path: str,
    ) -> str:
        """Generate comprehensive markdown report for model comparison."""
        best_model = comparison_df.loc[comparison_df["F1-Score"].idxmax(), "Model"]
        best_f1 = comparison_df["F1-Score"].max()
        performance_diff = abs(comparison_df["F1-Score"].iloc[0] - comparison_df["F1-Score"].iloc[1])

        class_dist = dataset_info["class_distribution"]
        imbalance_ratio = max(class_dist.values()) / min(class_dist.values())

        if comparison_df.iloc[0]["Model"] == "BERT-Chinese":
            bert_metrics = comparison_df.iloc[0]
            roberta_metrics = comparison_df.iloc[1]
        else:
            bert_metrics = comparison_df.iloc[1]
            roberta_metrics = comparison_df.iloc[0]

        report_content = f"""# BERT vs RoBERTa Chinese Text Classification Comparison Report

## Experiment Overview

**Experiment Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Task**: Binary Text Classification for Chinese Text  
**Dataset Size**: {dataset_info['total_samples']} samples  
**Models Compared**: BERT-Chinese vs RoBERTa-Chinese  

## Dataset Statistics

### Class Distribution
- **Total Samples**: {dataset_info['total_samples']}
- **Label 0 (Positive Feedback)**: {class_dist[0]} ({class_dist[0]/dataset_info['total_samples']*100:.1f}%)
- **Label 1 (Negative/Constructive Feedback)**: {class_dist[1]} ({class_dist[1]/dataset_info['total_samples']*100:.1f}%)
- **Imbalance Ratio**: {imbalance_ratio:.2f}:1

### Data Split
- **Training Set**: {dataset_info['train_size']} samples (80%)
- **Test Set**: {dataset_info['test_size']} samples (20%)

### Class Imbalance Handling
- **Method**: Balanced class weights using scikit-learn
- **Class Weights**: {dataset_info['class_weights']}

## Training Configuration

- **Learning Rate**: {self.training_config['learning_rate']}
- **Batch Size**: {self.training_config['per_device_train_batch_size']} per device
- **Epochs**: {self.training_config['num_train_epochs']}
- **Weight Decay**: {self.training_config['weight_decay']}
- **Mixed Precision**: {self.training_config['fp16']}
- **Evaluation Strategy**: {self.training_config['evaluation_strategy']}
- **Random Seed**: {self.training_config['seed']}

## Results Summary

| Metric | BERT-Chinese | RoBERTa-Chinese | Difference |
|--------|--------------|-----------------|------------|
| **Accuracy** | {bert_metrics['Accuracy']:.4f} | {roberta_metrics['Accuracy']:.4f} | {roberta_metrics['Accuracy'] - bert_metrics['Accuracy']:+.4f} |
| **Precision** | {bert_metrics['Precision']:.4f} | {roberta_metrics['Precision']:.4f} | {roberta_metrics['Precision'] - bert_metrics['Precision']:+.4f} |
| **Recall** | {bert_metrics['Recall']:.4f} | {roberta_metrics['Recall']:.4f} | {roberta_metrics['Recall'] - bert_metrics['Recall']:+.4f} |
| **F1-Score** | {bert_metrics['F1-Score']:.4f} | {roberta_metrics['F1-Score']:.4f} | {roberta_metrics['F1-Score'] - bert_metrics['F1-Score']:+.4f} |

## Key Findings

- **Best Performing Model**: {best_model}
- **Best F1-Score**: {best_f1:.4f}
- **Performance Gap**: {performance_diff:.4f} F1 points

## Publication Figures Generated

- **figure_transformer_model_comparison_panel.(png/pdf)**: 2x2 metrics panel
- **figure_transformer_accuracy.(png/pdf)**: standalone accuracy figure
- **figure_transformer_precision.(png/pdf)**: standalone precision figure
- **figure_transformer_recall.(png/pdf)**: standalone recall figure
- **figure_transformer_f1_score.(png/pdf)**: standalone F1-score figure
- **figure_transformer_confusion_matrices_panel.(png/pdf)**: side-by-side confusion matrices
- **figure_transformer_confusion_matrix_bert_chinese.(png/pdf)**: standalone BERT confusion matrix
- **figure_transformer_confusion_matrix_roberta_chinese.(png/pdf)**: standalone RoBERTa confusion matrix

---
*Report generated automatically on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        log_info(f"Comprehensive markdown report saved as '{report_path}'")
        return report_path

    def run(self) -> None:
        """Execute end-to-end training, evaluation, plotting, and reporting."""
        self._load_and_prepare_data()

        for model_name, config in self.models_config.items():
            self._train_one_model(model_name, config)

        log_info(f"\n{'=' * 60}")
        log_info("MODEL COMPARISON SUMMARY")
        log_info(f"{'=' * 60}")

        comparison_df = self._build_comparison_df()
        log_info(f"\n{comparison_df.round(4)}")

        best_model = comparison_df.loc[comparison_df["F1-Score"].idxmax(), "Model"]
        log_info(f"\nBest performing model: {best_model}")

        self._plot_model_comparison(comparison_df)
        self._plot_confusion_matrices()
        self._log_classification_reports()

        dataset_info = self._build_dataset_info(
            self.data,
            self.train_data,
            self.test_data,
            self.class_weights,
        )

        report_file = self.generate_markdown_report(
            comparison_df,
            dataset_info,
            report_path=str(self.result_dirs["reports"] / "BERT_RoBERTa_Comparison_Report.md"),
        )

        artifacts_dir = self.result_dirs["artifacts"]
        with open(artifacts_dir / "bert_roberta_training_info.json", "w", encoding="utf-8") as f:
            json.dump(self.training_artifacts, f, indent=2, ensure_ascii=False, default=str)

        comparison_df.to_csv(artifacts_dir / "bert_roberta_metrics_summary.csv", index=False, encoding="utf-8")

        log_info("Training completed!")
        log_info(f"Best model: {best_model}")
        log_info("Models saved in respective directories")
        log_info(f"Visualizations saved in '{self.result_dirs['figures']}'")
        log_info(f"Comprehensive report saved as '{report_file}'")
        log_info(f"Training and test prediction artifacts saved in '{artifacts_dir}'")


def main() -> int:
    pipeline = TransformerComparisonPipeline()
    pipeline.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
