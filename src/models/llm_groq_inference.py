"""Groq LLM inference pipeline for zero-shot and few-shot comment classification.

This module handles:
- Groq API interaction for LLM-based predictions
- Binary label parsing from LLM outputs with fallback strategies
- Configurable model selection via environment variables
- Results saved to results/llm/
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    plt = None
    sns = None

try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def get_project_root() -> Path:
    """Resolve project root from env or repository layout."""
    env_root = os.getenv("PROJECT_ROOT", "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    
    # Search for project root by finding Comment-Classification directory markers
    current = Path(__file__).resolve().parent
    while current != current.parent:
        # Check if this looks like the project root
        if (current / "data").exists() and (current / "src" / "models").exists():
            return current
        current = current.parent
    
    # Fallback to parents[2]
    return Path(__file__).resolve().parents[2]


def _load_env_file() -> None:
    """Load simple KEY=VALUE pairs from project .env into process environment."""
    env_path = get_project_root() / ".env"
    if not env_path.exists():
        return

    project_root = get_project_root()
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        
        # Resolve relative paths with respect to project root
        if key == "DATA_FILE" and value.startswith("./"):
            value = str((project_root / value).resolve())
        elif key == "DATA_FILE" and not Path(value).is_absolute():
            value = str((project_root / value).resolve())
        
        if key and key not in os.environ:
            os.environ[key] = value


def _get_default_model_names() -> List[str]:
    """Get LLM model names from environment or use hardcoded defaults."""
    _load_env_file()
    env_models = os.getenv("LLM_MODEL_NAMES", "").strip()
    if env_models:
        return [m.strip() for m in env_models.split(",") if m.strip()]
    return ["openai/gpt-oss-20b", "llama-3.3-70b-versatile"]


def _get_model_alias_map() -> Dict[str, str]:
    """Build model alias map from environment configuration."""
    _load_env_file()
    models = _get_default_model_names()
    return {model: model for model in models}


DEFAULT_MODEL_NAMES = _get_default_model_names()
MODEL_ALIAS_MAP = _get_model_alias_map()


def parse_llm_label(content: str) -> Optional[int]:
    """Parse binary label (0/1) from LLM output text."""
    text = str(content).strip().lower()

    if text in {"0", "1"}:
        return int(text)

    if "label: 0" in text or "predict: 0" in text or "class 0" in text:
        return 0
    if "label: 1" in text or "predict: 1" in text or "class 1" in text:
        return 1

    # JSON-like outputs: {"label": 0}, label=1, predicted_label: 0
    json_label = re.search(r'"?(label|predicted_label|class|prediction)"?\s*[:=]\s*([01])\b', text)
    if json_label:
        return int(json_label.group(2))

    # Standalone number token anywhere in the response.
    number_token = re.search(r'\b([01])\b', text)
    if number_token:
        return int(number_token.group(1))

    for ch in text:
        if ch in {"0", "1"}:
            return int(ch)
    return None


def _retry_parse_with_strict_prompt(client, model_name: str, raw_text: str) -> Optional[int]:
    """Ask model to normalize its own prior answer into strict 0/1 format."""
    repair_prompt = (
        "Convert the following classifier answer into one strict label only.\n"
        "Return only one character: 0 or 1.\n"
        "Do not add any other text.\n\n"
        f"Answer to normalize:\n{raw_text}\n\n"
        "Normalized label:"
    )

    response = client.chat.completions.create(
        model=model_name,
        temperature=0,
        max_tokens=4,
        messages=[{"role": "user", "content": repair_prompt}],
    )
    repaired = response.choices[0].message.content
    return parse_llm_label(repaired)


def build_zero_shot_prompt(text):
    """Build zero-shot prompt for binary text classification."""
    return (
        "You are a strict binary text classifier for Chinese comments.\n"
        "Task: classify the comment into one label only.\n"
        "Label definitions:\n"
        "- 0: No-meaningful comment\n"
        "- 1: Meaningful comment\n"
        "Output format: only a single character, either 0 or 1.\n\n"
        f"Comment:\n{text}\n\n"
        "Answer:"
    )


def build_few_shot_prompt(text, examples):
    """Build few-shot prompt with labeled examples."""
    example_blocks = []
    for i, ex in enumerate(examples, start=1):
        example_blocks.append(
            f"Example {i}:\n"
            f"Comment: {ex['text']}\n"
            f"Label: {int(ex['label'])}\n"
        )

    examples_text = "\n".join(example_blocks)
    return (
        "You are a strict binary text classifier for Chinese comments.\n"
        "Task: classify the comment into one label only.\n"
        "Label definitions:\n"
        "- 0: No-meaningful comment\n"
        "- 1: Meaningful comment\n"
        "Output format: only a single character, either 0 or 1.\n\n"
        "Labeled examples:\n"
        f"{examples_text}\n"
        f"Comment: {text}\n"
        "Label:"
    )


def select_few_shot_examples(train_texts, train_labels, max_per_class=3):
    """Select simple balanced examples from training split for few-shot prompting."""
    examples = []
    for label in [0, 1]:
        idx = np.where(train_labels == label)[0][:max_per_class]
        for i in idx:
            examples.append({"text": str(train_texts[i]), "label": int(train_labels[i])})
    return examples


def _get_groq_api_key() -> Optional[str]:
    """Read Groq API key from supported environment variable names."""
    _load_env_file()
    return os.getenv("GROQ_API_KEY") or os.getenv("Groq_API_KEY")


def _resolve_model_name(requested_model: str) -> str:
    """Resolve user-facing model name to provider model name."""
    return MODEL_ALIAS_MAP.get(requested_model, requested_model)


def _candidate_model_names(requested_model: str) -> List[str]:
    """Return ordered model ID candidates to maximize compatibility across Groq accounts."""
    base = requested_model.strip()
    alias = _resolve_model_name(base)
    candidates = [alias, base, base.lower()]

    # Deduplicate while preserving order.
    seen = set()
    ordered = []
    for c in candidates:
        if c and c not in seen:
            ordered.append(c)
            seen.add(c)
    return ordered


def _safe_model_name(model_name: str) -> str:
    """Build a filesystem-safe suffix for artifact file names."""
    return model_name.lower().replace("/", "_").replace("-", "_").replace(".", "_")


def _request_label_prediction(client, model_name: str, prompt: str) -> tuple[str, Optional[int]]:
    """Send one prompt to the model and return raw text + parsed binary label."""
    response = client.chat.completions.create(
        model=model_name,
        temperature=0,
        max_tokens=8,
        messages=[{"role": "user", "content": prompt}],
    )
    response_text = response.choices[0].message.content
    parsed = parse_llm_label(response_text)
    if parsed is None:
        try:
            parsed = _retry_parse_with_strict_prompt(client, model_name, response_text)
        except Exception:
            parsed = None
    return response_text, parsed


def get_llm_results_dir() -> Path:
    """Create and return standardized output directory for LLM artifacts."""
    project_root = get_project_root()
    out_dir = project_root / "results" / "llm"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _calc_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate binary classification metrics and invalid response count."""
    valid_df = df[df["predicted_label"].isin([0, 1])]
    if len(valid_df) == 0:
        return {
            "evaluated_samples": 0,
            "invalid_predictions": int(len(df)),
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1_score": np.nan,
        }

    y_true = valid_df["true_label"].values
    y_pred = valid_df["predicted_label"].values
    return {
        "evaluated_samples": int(len(valid_df)),
        "invalid_predictions": int(len(df) - len(valid_df)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="binary", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="binary", zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
    }


def _run_inference_mode(
    client,
    model_name: str,
    texts_eval: np.ndarray,
    labels_eval: np.ndarray,
    mode: str,
    few_shot_examples: Optional[List[Dict[str, object]]] = None,
    progress_desc: Optional[str] = None,
) -> pd.DataFrame:
    """Run one inference mode (`zero_shot` or `few_shot`) and return rows as DataFrame."""
    rows = []

    iterable = tqdm(texts_eval, desc=progress_desc, total=len(texts_eval)) if progress_desc else texts_eval
    for i, text in enumerate(iterable):
        text_str = str(text)
        if mode == "few_shot":
            prompt = build_few_shot_prompt(text_str, few_shot_examples or [])
        else:
            prompt = build_zero_shot_prompt(text_str)

        raw_text, parsed_label = _request_label_prediction(client, model_name, prompt)
        rows.append(
            {
                "sample_index": i,
                "text": text_str,
                "true_label": int(labels_eval[i]),
                "predicted_label": int(parsed_label) if parsed_label is not None else -1,
                "raw_response": str(raw_text),
            }
        )

    return pd.DataFrame(rows)


class GroqLLMInferenceRunner:
    """Object-oriented wrapper around Groq LLM inference helpers."""

    def __init__(self, logger=None):
        self.logger = logger

    def run(
        self,
        test_texts,
        test_labels,
        train_texts,
        train_labels,
        artifacts_dir,
        model_names=None,
        max_samples=200,
    ):
        """Execute the existing Groq inference flow through a class interface."""
        return run_groq_llm_inference(
            test_texts=test_texts,
            test_labels=test_labels,
            train_texts=train_texts,
            train_labels=train_labels,
            artifacts_dir=artifacts_dir,
            logger=self.logger,
            model_names=model_names,
            max_samples=max_samples,
        )


def run_groq_llm_inference(
    test_texts,
    test_labels,
    train_texts,
    train_labels,
    artifacts_dir,
    logger=None,
    model_names=None,
    max_samples=200,
):
    """Run Groq zero-shot and few-shot inference for one or more models."""
    if Groq is None:
        if logger is not None:
            logger.warning("Groq SDK is not installed. Skipping LLM inference.")
        return None

    api_key = _get_groq_api_key()
    if not api_key:
        if logger is not None:
            logger.warning("GROQ_API_KEY (or Groq_API_KEY) not found. Skipping LLM inference.")
        return None

    client = Groq(api_key=api_key)

    if model_names is None:
        model_names = DEFAULT_MODEL_NAMES

    num_samples = min(max_samples, len(test_texts))
    texts_eval = np.array(test_texts[:num_samples], dtype=object)
    labels_eval = np.array(test_labels[:num_samples], dtype=int)

    few_shot_examples = select_few_shot_examples(
        np.array(train_texts, dtype=object),
        np.array(train_labels, dtype=int),
    )

    all_metrics = {"max_samples": int(num_samples), "models": {}}

    successful_models = 0
    all_metrics["failed_models"] = {}

    for requested_model in model_names:
        model_success = False
        last_error = None

        for candidate in _candidate_model_names(requested_model):
            if logger is not None:
                logger.info(f"Running Groq inference with model: {requested_model} -> {candidate}")

            try:
                zero_df = _run_inference_mode(
                    client=client,
                    model_name=candidate,
                    texts_eval=texts_eval,
                    labels_eval=labels_eval,
                    mode="zero_shot",
                    progress_desc=f"Groq zero-shot ({requested_model})",
                )
                few_df = _run_inference_mode(
                    client=client,
                    model_name=candidate,
                    texts_eval=texts_eval,
                    labels_eval=labels_eval,
                    mode="few_shot",
                    few_shot_examples=few_shot_examples,
                    progress_desc=f"Groq few-shot ({requested_model})",
                )
            except Exception as exc:
                last_error = str(exc)
                if logger is not None:
                    logger.warning(f"Model candidate failed for {requested_model}: {candidate} -> {exc}")
                continue

            safe_name = _safe_model_name(requested_model)
            zero_df.to_csv(os.path.join(artifacts_dir, f"groq_{safe_name}_zero_shot_predictions.csv"), index=False, encoding="utf-8")
            few_df.to_csv(os.path.join(artifacts_dir, f"groq_{safe_name}_few_shot_predictions.csv"), index=False, encoding="utf-8")

            all_metrics["models"][requested_model] = {
                "resolved_model": candidate,
                "zero_shot": _calc_metrics(zero_df),
                "few_shot": _calc_metrics(few_df),
            }
            successful_models += 1
            model_success = True
            break

        if not model_success:
            all_metrics["failed_models"][requested_model] = {
                "error": last_error or "unknown_error",
            }
            if logger is not None:
                logger.error(f"All model candidates failed for {requested_model}")

    if successful_models == 0:
        if logger is not None:
            logger.error("No requested Groq models were available.")
        return None

    with open(os.path.join(artifacts_dir, "groq_llm_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    if logger is not None:
        logger.info("Groq multi-model zero-shot and few-shot inference completed")
    
    # Generate visualizations
    _create_llm_visualizations(artifacts_dir, all_metrics, logger)
    
    return all_metrics


def _setup_professional_style() -> None:
    """Apply professional publication-oriented plotting style."""
    if plt is None or sns is None:
        return
    
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
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': ':',
    })
    sns.set_palette("husl")


def _save_figure_multi_format(fig: plt.Figure, output_path: Path, filename_stem: str, formats: tuple = ("png", "pdf")) -> None:
    """Save figure in multiple formats with consistent DPI."""
    if plt is None:
        return
    
    output_path.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        filepath = output_path / f"{filename_stem}.{fmt}"
        fig.savefig(filepath, dpi=600, bbox_inches='tight', format=fmt)


def _create_llm_visualizations(artifacts_dir: str, metrics_dict: Dict, logger=None) -> None:
    """Generate confusion matrices and metrics comparison figures for LLM models."""
    if plt is None or sns is None:
        if logger:
            logger.warning("matplotlib/seaborn not available. Skipping LLM visualizations.")
        return
    
    _setup_professional_style()
    output_dir = Path(artifacts_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    if "models" not in metrics_dict:
        if logger:
            logger.warning("No models in metrics dict. Skipping visualizations.")
        return
    
    models = metrics_dict["models"]
    
    # 1. Create confusion matrices for each model and mode
    for model_name, model_data in models.items():
        safe_name = model_name.lower().replace("/", "_").replace("-", "_").replace(".", "_")
        
        for mode in ["zero_shot", "few_shot"]:
            csv_path = output_dir / f"groq_{safe_name}_{mode}_predictions.csv"
            if not csv_path.exists():
                if logger:
                    logger.debug(f"CSV not found: {csv_path}")
                continue
            
            # Load predictions
            pred_df = pd.read_csv(csv_path, encoding="utf-8")
            valid_df = pred_df[pred_df["predicted_label"].isin([0, 1])]
            
            if len(valid_df) > 0:
                y_true = valid_df["true_label"].values
                y_pred = valid_df["predicted_label"].values
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                
                # Plot confusion matrix
                fig, ax = plt.subplots(figsize=(6, 5))
                im = ax.imshow(cm, cmap='Blues', aspect='auto')
                ax.set_title(f"{model_name}\n{mode.replace('_', ' ').title()}", fontsize=13)
                ax.set_xlabel("Predicted", fontsize=12)
                ax.set_ylabel("True", fontsize=12)
                
                # Add text annotations
                for i in range(2):
                    for j in range(2):
                        text = ax.text(j, i, cm[i, j],
                                     ha="center", va="center",
                                     color="white" if cm[i, j] > cm.max() / 2 else "black",
                                     fontsize=14, fontweight='bold')
                
                ax.set_xticks([0, 1])
                ax.set_yticks([0, 1])
                ax.set_xticklabels(['No-meaningful', 'Meaningful'])
                ax.set_yticklabels(['No-meaningful', 'Meaningful'])
                
                plt.colorbar(im, ax=ax, label="Count")
                plt.tight_layout()
                
                filename = f"llm_{safe_name}_{mode}_confusion_matrix"
                _save_figure_multi_format(fig, figures_dir, filename, ("png", "pdf"))
                plt.close(fig)
                
                if logger:
                    logger.info(f"Saved confusion matrix: {figures_dir / (filename + '.png')}")
    
    # 2. Create metrics comparison bar chart (zero-shot vs few-shot)
    models_with_valid_metrics = []
    zero_shot_metrics = []
    few_shot_metrics = []
    
    for model_name, model_data in models.items():
        zero_metrics = model_data.get("zero_shot", {})
        few_metrics = model_data.get("few_shot", {})
        
        if (zero_metrics.get("evaluated_samples", 0) > 0 or few_metrics.get("evaluated_samples", 0) > 0):
            models_with_valid_metrics.append(model_name)
            zero_shot_metrics.append(zero_metrics)
            few_shot_metrics.append(few_metrics)
    
    if models_with_valid_metrics:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        metrics_names = ['accuracy', 'precision', 'recall', 'f1_score']
        x = np.arange(len(models_with_valid_metrics))
        width = 0.08
        
        for i, metric in enumerate(metrics_names):
            zero_values = [float(m.get(metric, 0)) if not np.isnan(float(m.get(metric, 0))) else 0 
                          for m in zero_shot_metrics]
            few_values = [float(m.get(metric, 0)) if not np.isnan(float(m.get(metric, 0))) else 0 
                         for m in few_shot_metrics]
            
            ax.bar(x + i * width - 1.5 * width, zero_values, width, label=f"{metric.replace('_', ' ').title()} (Zero-shot)")
            ax.bar(x + i * width + 1.5 * width, few_values, width, label=f"{metric.replace('_', ' ').title()} (Few-shot)")
        
        # Simplify legend by removing duplicates
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=9)
        
        ax.set_xlabel("Model", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("LLM Model Performance: Zero-shot vs Few-shot", fontsize=13)
        ax.set_xticks(x + width)
        ax.set_xticklabels(models_with_valid_metrics, rotation=15, ha='right')
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        _save_figure_multi_format(fig, figures_dir, "llm_metrics_comparison", ("png", "pdf"))
        plt.close(fig)
        
        if logger:
            logger.info(f"Saved metrics comparison: {figures_dir / 'llm_metrics_comparison.png'}")
    
    # 3. Create F1-score comparison chart
    f1_data = []
    for model_name, model_data in models.items():
        zero_f1 = model_data.get("zero_shot", {}).get("f1_score", 0)
        few_f1 = model_data.get("few_shot", {}).get("f1_score", 0)
        
        if not np.isnan(float(zero_f1)):
            f1_data.append({"Model": f"{model_name}\n(Zero-shot)", "F1-score": float(zero_f1)})
        if not np.isnan(float(few_f1)):
            f1_data.append({"Model": f"{model_name}\n(Few-shot)", "F1-score": float(few_f1)})
    
    if f1_data:
        f1_df = pd.DataFrame(f1_data).sort_values("F1-score", ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(f1_df["Model"], f1_df["F1-score"], color="#1f77b4", edgecolor="black")
        
        ax.set_title("LLM F1-Score Comparison", fontsize=13)
        ax.set_ylabel("F1-Score", fontsize=12)
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=10)
        
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        _save_figure_multi_format(fig, figures_dir, "llm_f1_score_comparison", ("png", "pdf"))
        plt.close(fig)
        
        if logger:
            logger.info(f"Saved F1-score comparison: {figures_dir / 'llm_f1_score_comparison.png'}")


def _load_default_train_test_split():
    """Load DATA_FILE and create a reproducible train/test split for standalone LLM runs."""
    _load_env_file()
    project_root = get_project_root()
    # Get from environment (already resolved by _load_env_file) or use default
    data_file_str = os.getenv("DATA_FILE")
    if not data_file_str:
        data_file_str = str(project_root / "data" / "Dataset.csv")
    data_file = Path(data_file_str).resolve()
    data = pd.read_csv(str(data_file), encoding="utf-8")

    if "text" not in data.columns or "label" not in data.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns")

    data = data.copy()
    data["text"] = data["text"].astype(str)
    data["label"] = pd.to_numeric(data["label"], errors="coerce")
    data = data.dropna(subset=["label"]) 
    data["label"] = data["label"].astype(int)

    # Accept {1,2} labels and map to internal {1,0}.
    if set(data["label"].unique()).issubset({1, 2}):
        data["label"] = data["label"].replace({2: 0})

    data = data[data["label"].isin([0, 1])]

    texts = data["text"].to_numpy(dtype=object)
    labels = data["label"].to_numpy(dtype=int)

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
        shuffle=True,
    )

    return train_texts, test_texts, train_labels, test_labels


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    project_root = get_project_root()
    artifacts_dir = get_llm_results_dir()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    try:
        train_texts, test_texts, train_labels, test_labels = _load_default_train_test_split()
        metrics = run_groq_llm_inference(
            test_texts=test_texts,
            test_labels=test_labels,
            train_texts=train_texts,
            train_labels=train_labels,
            artifacts_dir=str(artifacts_dir),
            logger=logger,
            model_names=DEFAULT_MODEL_NAMES,
            max_samples=200,
        )
    except Exception as exc:
        logger.error(f"Standalone LLM inference failed: {exc}")
        return 1

    if metrics is None:
        logger.error("LLM inference did not run (missing Groq SDK or API key).")
        return 1

    logger.info("Standalone LLM inference completed successfully.")
    
    # Generate visualizations
    _create_llm_visualizations(str(artifacts_dir), metrics, logger)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
