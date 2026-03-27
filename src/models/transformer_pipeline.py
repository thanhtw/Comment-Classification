"""Transformer pipeline for binary text classification using BERT/RoBERTa.

This module implements:
- 10-fold stratified cross-validation with WeightedTrainer
- BERT and RoBERTa model fine-tuning
- Class weight balancing and gradient accumulation
- Best-fold selection and comprehensive reporting
- Publication-quality visualizations and metrics
"""

import json
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
)

# Import professional figure utilities
from src.utils.figure_utils import (
    export_fold_metrics_csv,
    plot_confusion_matrix_consistent,
    plot_fold_metrics_comparison,
    plot_metrics_panel,
    plot_loss_comparison,
    setup_professional_style,
    save_figure_multi_format
)

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

HOLDOUT_TEST_SIZE = 0.2


def log_info(message: str) -> None:
    logger.info(message)


def get_project_root() -> Path:
    env_root = os.getenv("PROJECT_ROOT", "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def get_results_dirs() -> dict:
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
    def __init__(self) -> None:
        self.result_dirs = get_results_dirs()
        project_root = get_project_root()
        self.data_file = Path(os.getenv("DATA_FILE", str(project_root / "data" / "Dataset.csv")))
        self.num_folds = 10

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
        self.fold_results = {name: [] for name in self.models_config.keys()}
        self.best_fold_records = {}
        self.heldout_predictions = {}
        self.figure_dpi = 600
        self.figure_formats = ("png", "pdf")

    @staticmethod
    def compute_metrics(eval_pred):
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
        class_weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
        return {i: weight for i, weight in enumerate(class_weights)}

    def _set_publication_style(self) -> None:
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
        for fmt in self.figure_formats:
            fig.savefig(self.result_dirs["figures"] / f"{filename_stem}.{fmt}", dpi=self.figure_dpi, bbox_inches="tight")

    def _load_data(self) -> None:
        self.data = pd.read_csv(self.data_file, encoding="utf-8")
        if "text" not in self.data.columns or "label" not in self.data.columns:
            raise ValueError("Dataset must contain 'text' and 'label' columns.")

        label_to_int = {label: idx for idx, label in enumerate(sorted(self.data["label"].unique()))}
        self.data["label"] = self.data["label"].map(label_to_int)

        log_info(f"Dataset loaded: {self.data.shape}")
        log_info(f"Label distribution:\n{self.data['label'].value_counts()}")

    def _train_one_model_on_fold(self, model_name: str, config: dict, fold_idx: int, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
        tokenizer = config["tokenizer_class"].from_pretrained(config["model_name"])
        model = config["model_class"].from_pretrained(
            config["model_name"],
            num_labels=2,
            hidden_dropout_prob=0.2,
            attention_probs_dropout_prob=0.1,
        )

        train_dataset = Dataset.from_pandas(train_df[["text", "label"]], preserve_index=False)
        val_dataset = Dataset.from_pandas(val_df[["text", "label"]], preserve_index=False)

        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=200)

        tokenized_train = train_dataset.map(tokenize_function, batched=True)
        tokenized_val = val_dataset.map(tokenize_function, batched=True)

        class_weights = self._calculate_class_weights(train_df["label"].values)
        training_args = TrainingArguments(
            output_dir=str(self.result_dirs["runs"] / f"{model_name.replace('-', '_')}_fold_{fold_idx}"),
            **self.training_config,
        )

        trainer = WeightedTrainer(
            class_weights=class_weights,
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            compute_metrics=self.compute_metrics,
        )

        log_info(f"Training {model_name} - fold {fold_idx}/{self.num_folds}")
        train_result = trainer.train()
        eval_result = trainer.evaluate()

        predictions = trainer.predict(tokenized_val)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = predictions.label_ids

        fold_row = {
            "fold": int(fold_idx),
            "accuracy": float(eval_result["eval_accuracy"]),
            "precision": float(eval_result["eval_precision"]),
            "recall": float(eval_result["eval_recall"]),
            "f1_score": float(eval_result["eval_f1"]),
            "loss": float(eval_result.get("eval_loss", np.nan)),
        }
        self.fold_results[model_name].append(fold_row)

        pred_df = pd.DataFrame(
            {
                "fold": fold_idx,
                "sample_index": np.arange(len(y_true)),
                "true_label": y_true.astype(int),
                "predicted_label": y_pred.astype(int),
                "model": model_name,
            }
        )
        pred_df.to_csv(
            self.result_dirs["artifacts"] / f"{model_name.replace('-', '_').lower()}_fold_{fold_idx}_predictions.csv",
            index=False,
            encoding="utf-8",
        )

        current_best = self.best_fold_records.get(model_name)
        if current_best is None or fold_row["f1_score"] > current_best["metrics"]["f1_score"]:
            best_dir = self.result_dirs["models"] / f"{model_name.replace('-', '_')}_best_fold"
            model.save_pretrained(str(best_dir))
            tokenizer.save_pretrained(str(best_dir))
            self.best_fold_records[model_name] = {
                "pipeline": "transformer",
                "model": model_name,
                "best_fold": int(fold_idx),
                "selection_metric": "f1_score",
                "metrics": {
                    "accuracy": fold_row["accuracy"],
                    "precision": fold_row["precision"],
                    "recall": fold_row["recall"],
                    "f1_score": fold_row["f1_score"],
                    "loss": fold_row["loss"],
                },
                "y_true": y_true.tolist(),
                "y_pred": y_pred.tolist(),
                "train_metrics": train_result.metrics,
                "eval_metrics": eval_result,
                "model_dir": str(best_dir),
            }

        del trainer, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _run_cross_validation(self) -> None:
        skf = StratifiedKFold(n_splits=self.num_folds, shuffle=True, random_state=42)
        X = self.train_data["text"].to_numpy(dtype=object)
        y = self.train_data["label"].to_numpy(dtype=int)

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
            train_df = self.train_data.iloc[train_idx].copy()
            val_df = self.train_data.iloc[val_idx].copy()
            for model_name, config in self.models_config.items():
                self._train_one_model_on_fold(model_name, config, fold_idx, train_df, val_df)

    def _prepare_holdout_split(self) -> None:
        """Create a shared held-out test split for confusion-matrix comparability across pipelines."""
        train_df, test_df = train_test_split(
            self.data,
            test_size=HOLDOUT_TEST_SIZE,
            random_state=42,
            stratify=self.data["label"],
            shuffle=True,
        )
        self.train_data = train_df.reset_index(drop=True)
        self.test_data = test_df.reset_index(drop=True)
        log_info(f"Transformer train samples: {len(self.train_data)}")
        log_info(f"Transformer held-out test samples: {len(self.test_data)}")

    def _evaluate_best_models_on_heldout_test(self) -> None:
        """Evaluate best-fold checkpoints on the shared held-out test split."""
        self.heldout_predictions = {}

        for model_name, rec in self.best_fold_records.items():
            best_dir = Path(rec["model_dir"])
            model_cfg = self.models_config[model_name]
            tokenizer = model_cfg["tokenizer_class"].from_pretrained(str(best_dir))
            model = model_cfg["model_class"].from_pretrained(str(best_dir))

            test_dataset = Dataset.from_pandas(self.test_data[["text", "label"]], preserve_index=False)

            def tokenize_function(examples):
                return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=200)

            tokenized_test = test_dataset.map(tokenize_function, batched=True)

            test_args = TrainingArguments(
                output_dir=str(self.result_dirs["runs"] / f"{model_name.replace('-', '_')}_heldout_eval"),
                per_device_eval_batch_size=self.training_config["per_device_eval_batch_size"],
                dataloader_drop_last=False,
                fp16=self.training_config["fp16"],
                report_to=[],
            )

            trainer = Trainer(
                model=model,
                args=test_args,
                compute_metrics=self.compute_metrics,
            )

            pred_output = trainer.predict(tokenized_test)
            y_pred = np.argmax(pred_output.predictions, axis=1).astype(int)
            y_true = pred_output.label_ids.astype(int)

            self.heldout_predictions[model_name] = {
                "y_true": y_true,
                "y_pred": y_pred,
            }

            pd.DataFrame(
                {
                    "sample_index": np.arange(len(y_true)),
                    "true_label": y_true,
                    "predicted_label": y_pred,
                    "model": model_name,
                }
            ).to_csv(
                self.result_dirs["artifacts"] / f"{model_name.replace('-', '_').lower()}_heldout_test_predictions.csv",
                index=False,
                encoding="utf-8",
            )

            del trainer, model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _build_best_fold_comparison_df(self) -> pd.DataFrame:
        rows = []
        for model_name, rec in self.best_fold_records.items():
            rows.append(
                {
                    "Model": model_name,
                    "Best Fold": rec["best_fold"],
                    "Accuracy": rec["metrics"]["accuracy"],
                    "Precision": rec["metrics"]["precision"],
                    "Recall": rec["metrics"]["recall"],
                    "F1-Score": rec["metrics"]["f1_score"],
                }
            )
        return pd.DataFrame(rows).sort_values("F1-Score", ascending=False).reset_index(drop=True)

    def _plot_model_comparison(self, comparison_df: pd.DataFrame) -> None:
        """Create professional comparative visualizations of model performance."""
        self._set_publication_style()
        
        # Export per-fold metrics to CSV for all models
        output_path = self.result_dirs["figures"]
        export_fold_metrics_csv(self.fold_results, output_path, "transformer_fold_metrics")
        
        # Create separate metrics figures
        plot_metrics_panel(self.fold_results, output_path, "transformer_metrics_panel", separate=True)
        
        # Create loss comparison if available
        loss_fig = plot_loss_comparison(self.fold_results, output_path, "transformer_loss_comparison")
        
        # Legacy best-fold comparison for backward compatibility
        metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
        colors = ["#1f77b4", "#ff7f0e"]

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        for i, metric in enumerate(metrics):
            ax = axes[i // 2, i % 2]
            bars = ax.bar(comparison_df["Model"], comparison_df[metric], color=colors, width=0.6, edgecolor='black', alpha=0.8)
            ax.set_title(f"Best-Fold {metric} Comparison", fontsize=12, fontweight='bold')
            ax.set_ylabel(metric, fontsize=11)
            ax.set_ylim(0, 1)
            ax.grid(True, axis="y", alpha=0.25)
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.01, f"{h:.4f}", ha="center", va="bottom", fontsize=10, fontweight='bold')
        plt.suptitle("Transformer Model Performance (Best Fold)", fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        self._save_figure_multi_format(fig, "figure_transformer_best_fold_comparison_panel")
        plt.close(fig)

    def _plot_confusion_matrices(self) -> None:
        """Create individual confusion matrix visualizations on held-out test split."""
        model_names = list(self.heldout_predictions.keys())

        for model_name in model_names:
            rec = self.heldout_predictions[model_name]
            filename_stem = f"transformer_confusion_matrix_{model_name.lower()}_heldout_test"
            self.result_dirs["figures"].mkdir(parents=True, exist_ok=True)

            fig = plot_confusion_matrix_consistent(
                y_true=rec["y_true"],
                y_pred=rec["y_pred"],
                title=f"{model_name} - Confusion Matrix (Held-out Test)",
                output_path=self.result_dirs["figures"],
                filename_stem=filename_stem,
                class_labels=["No-Meaningful", "Meaningful"],
                formats=self.figure_formats,
            )
            plt.close(fig)

    def _log_classification_reports(self) -> None:
        for model_name, rec in self.heldout_predictions.items():
            log_info(f"\n{model_name} - Held-out Test Classification Report:")
            log_info("-" * 60)
            log_info(
                classification_report(
                    rec["y_true"],
                    rec["y_pred"],
                    target_names=["No-Meaningful", "Meaningful"],
                )
            )

    def generate_markdown_report(self, comparison_df: pd.DataFrame, report_path: str) -> str:
        best_model = comparison_df.loc[comparison_df["F1-Score"].idxmax(), "Model"]
        best_f1 = comparison_df[comparison_df["Model"] == best_model]["F1-Score"].iloc[0]

        table_rows = []
        for _, row in comparison_df.iterrows():
            table_rows.append(
                f"| {row['Model']} | {int(row['Best Fold'])} | {row['Accuracy']:.4f} | {row['Precision']:.4f} | {row['Recall']:.4f} | {row['F1-Score']:.4f} |"
            )

        report_content = f"""# Transformer Models 10-Fold Cross-Validation Report

## Experiment Overview

- **Task**: Binary Text Classification for Chinese Text
- **Models**: BERT-Chinese, RoBERTa-Chinese
- **Validation Scheme**: {self.num_folds}-fold stratified cross-validation
- **Selection Rule**: best fold per model by validation F1-score

## Best-Fold Metrics by Model

| Model | Best Fold | Accuracy | Precision | Recall | F1-Score |
|------|-----------|----------|-----------|--------|----------|
{os.linesep.join(table_rows)}

## Key Findings

- **Best Transformer Model**: {best_model}
- **Best F1-Score**: {best_f1:.4f}

## Publication Figures Generated

- **figure_transformer_best_fold_comparison_panel.(png/pdf)**
- **figure_transformer_best_fold_confusion_matrices.(png/pdf)**

---
*Report generated automatically on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        return report_path

    def _save_artifacts(self, comparison_df: pd.DataFrame, report_file: str) -> None:
        artifacts_dir = self.result_dirs["artifacts"]

        fold_results_path = artifacts_dir / "transformer_fold_results.json"
        with open(fold_results_path, "w", encoding="utf-8") as f:
            json.dump(self.fold_results, f, indent=2, ensure_ascii=False)

        best_fold_summary = {
            model_name: {
                "pipeline": rec["pipeline"],
                "model": rec["model"],
                "best_fold": rec["best_fold"],
                "selection_metric": rec["selection_metric"],
                "metrics": rec["metrics"],
            }
            for model_name, rec in self.best_fold_records.items()
        }

        with open(artifacts_dir / "transformer_best_fold_summary.json", "w", encoding="utf-8") as f:
            json.dump(best_fold_summary, f, indent=2, ensure_ascii=False)

        comparison_df.to_csv(artifacts_dir / "transformer_best_fold_metrics_summary.csv", index=False, encoding="utf-8")

        training_info = {
            "experiment": "TransformerComparisonPipeline",
            "num_folds": self.num_folds,
            "total_samples": int(len(self.data)),
            "cv_training_samples": int(len(self.train_data)),
            "heldout_test_samples": int(len(self.test_data)),
            "fold_results_file": str(fold_results_path),
            "best_fold_summary": best_fold_summary,
            "report_file": report_file,
        }
        with open(artifacts_dir / "transformer_training_info.json", "w", encoding="utf-8") as f:
            json.dump(training_info, f, indent=2, ensure_ascii=False)

    def run(self) -> None:
        self._load_data()
        self._prepare_holdout_split()
        self._run_cross_validation()
        self._evaluate_best_models_on_heldout_test()

        comparison_df = self._build_best_fold_comparison_df()
        log_info("\nTransformer best-fold summary:")
        log_info(f"\n{comparison_df}")

        self._plot_model_comparison(comparison_df)
        self._plot_confusion_matrices()
        self._log_classification_reports()

        report_file = self.generate_markdown_report(
            comparison_df,
            report_path=str(self.result_dirs["reports"] / "BERT_RoBERTa_Comparison_Report.md"),
        )
        self._save_artifacts(comparison_df, report_file)

        log_info("Transformer 10-fold pipeline completed.")
        log_info(f"Report saved to: {report_file}")


def main() -> int:
    pipeline = TransformerComparisonPipeline()
    pipeline.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
