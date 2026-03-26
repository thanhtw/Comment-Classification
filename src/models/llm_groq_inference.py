import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


DEFAULT_MODEL_NAMES = ["Qwen2.5-7B-Instruct", "llama-3.1-8b-instant"]
MODEL_ALIAS_MAP = {
    "Qwen2.5-7B-Instruct": "qwen/qwen2.5-7b-instruct",
    "qwen2.5-7b-instruct": "qwen/qwen2.5-7b-instruct",
    "llama-3.1-8b-instant": "llama-3.1-8b-instant",
}


def parse_llm_label(content: str) -> Optional[int]:
    """Parse binary label (0/1) from LLM output text."""
    text = str(content).strip().lower()

    if text in {"0", "1"}:
        return int(text)

    if "label: 0" in text or "predict: 0" in text or "class 0" in text:
        return 0
    if "label: 1" in text or "predict: 1" in text or "class 1" in text:
        return 1

    for ch in text:
        if ch in {"0", "1"}:
            return int(ch)
    return None


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
    return os.getenv("GROQ_API_KEY") or os.getenv("Groq_API_KEY")


def _resolve_model_name(requested_model: str) -> str:
    """Resolve user-facing model name to provider model name."""
    return MODEL_ALIAS_MAP.get(requested_model, requested_model)


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
    return response_text, parse_llm_label(response_text)


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

    for requested_model in model_names:
        resolved_model = _resolve_model_name(requested_model)
        if logger is not None:
            logger.info(f"Running Groq inference with model: {requested_model} -> {resolved_model}")

        zero_df = _run_inference_mode(
            client=client,
            model_name=resolved_model,
            texts_eval=texts_eval,
            labels_eval=labels_eval,
            mode="zero_shot",
            progress_desc=f"Groq zero-shot ({requested_model})",
        )
        few_df = _run_inference_mode(
            client=client,
            model_name=resolved_model,
            texts_eval=texts_eval,
            labels_eval=labels_eval,
            mode="few_shot",
            few_shot_examples=few_shot_examples,
            progress_desc=f"Groq few-shot ({requested_model})",
        )

        safe_name = _safe_model_name(requested_model)
        zero_df.to_csv(os.path.join(artifacts_dir, f"groq_{safe_name}_zero_shot_predictions.csv"), index=False, encoding="utf-8")
        few_df.to_csv(os.path.join(artifacts_dir, f"groq_{safe_name}_few_shot_predictions.csv"), index=False, encoding="utf-8")

        all_metrics["models"][requested_model] = {
            "resolved_model": resolved_model,
            "zero_shot": _calc_metrics(zero_df),
            "few_shot": _calc_metrics(few_df),
        }

    with open(os.path.join(artifacts_dir, "groq_llm_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    if logger is not None:
        logger.info("Groq multi-model zero-shot and few-shot inference completed")
    return all_metrics
