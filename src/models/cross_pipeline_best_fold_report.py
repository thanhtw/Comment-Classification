"""Cross-pipeline best-fold comparison and unified reporting.

This module:
- Collects best-fold metrics from all pipelines (ML, Deep, Transformer)
- Includes LLM results when available
- Generates unified CSV, figures, and markdown reports
- Provides publication-ready visualizations for comparison
"""

import json
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.config import get_project_root


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CrossPipelineBestFoldReporter:
    def __init__(self) -> None:
        self.project_root = get_project_root()
        self.results_root = self.project_root / "results"
        self.output_reports = self.results_root / "reports"
        self.output_figures = self.results_root / "figures"
        self.output_artifacts = self.results_root / "artifacts"

        for d in [self.output_reports, self.output_figures, self.output_artifacts]:
            d.mkdir(parents=True, exist_ok=True)

    def _load_json_if_exists(self, path: Path):
        if not path.exists():
            logger.warning(f"Missing artifact: {path}")
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _collect_best_fold_rows(self):
        rows = []

        ml_path = self.results_root / "machine_learning" / "artifacts" / "ml_best_fold_summary.json"
        deep_path = self.results_root / "deep_learning" / "artifacts" / "deep_learning_best_fold_summary.json"
        tr_path = self.results_root / "transformer" / "artifacts" / "transformer_best_fold_summary.json"
        llm_path = self.results_root / "llm" / "groq_llm_metrics.json"

        for path in [ml_path, deep_path, tr_path]:
            summary = self._load_json_if_exists(path)
            if not summary:
                continue
            for model_name, info in summary.items():
                metrics = info.get("metrics", {})
                rows.append(
                    {
                        "pipeline": info.get("pipeline", "unknown"),
                        "model": model_name,
                        "fold": int(info.get("best_fold", 0)),
                        "accuracy": float(metrics.get("accuracy", 0.0)),
                        "precision": float(metrics.get("precision", 0.0)),
                        "recall": float(metrics.get("recall", 0.0)),
                        "f1_score": float(metrics.get("f1_score", 0.0)),
                    }
                )

        llm_metrics = self._load_json_if_exists(llm_path)
        if llm_metrics is None:
            # Backward compatibility for runs saved under ML artifacts in older versions.
            legacy_llm_path = self.results_root / "llm" / "artifacts" / "groq_llm_metrics.json"
            llm_metrics = self._load_json_if_exists(legacy_llm_path)
        if llm_metrics and "models" in llm_metrics:
            for model_name, info in llm_metrics["models"].items():
                for mode in ["zero_shot", "few_shot"]:
                    m = info.get(mode, {})
                    if m.get("evaluated_samples", 0) <= 0:
                        continue
                    rows.append(
                        {
                            "pipeline": "llm",
                            "model": f"{model_name} ({mode})",
                            "fold": 0,
                            "accuracy": float(m.get("accuracy", 0.0)),
                            "precision": float(m.get("precision", 0.0)),
                            "recall": float(m.get("recall", 0.0)),
                            "f1_score": float(m.get("f1_score", 0.0)),
                        }
                    )

        return rows

    def _save_plot(self, df: pd.DataFrame) -> Path:
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams.update(
            {
                "font.family": "DejaVu Serif",
                "font.size": 10,
                "axes.spines.top": False,
                "axes.spines.right": False,
            }
        )

        df_plot = df.sort_values("f1_score", ascending=False).reset_index(drop=True)
        labels = [f"{r['pipeline']}:{r['model']}" for _, r in df_plot.iterrows()]

        fig, ax = plt.subplots(figsize=(14, 6))
        bars = ax.bar(labels, df_plot["f1_score"], color="#1f4e79")
        ax.set_title("Best-Fold F1 Score Comparison Across All Models")
        ax.set_ylabel("F1 Score")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=50)
        ax.grid(True, axis="y", alpha=0.25)

        for bar, value in zip(bars, df_plot["f1_score"]):
            ax.text(bar.get_x() + bar.get_width() / 2.0, value + 0.01, f"{value:.4f}", ha="center", va="bottom", fontsize=8)

        fig.tight_layout()
        out_png = self.output_figures / "cross_pipeline_best_fold_f1_comparison.png"
        out_pdf = self.output_figures / "cross_pipeline_best_fold_f1_comparison.pdf"
        fig.savefig(out_png, dpi=600, bbox_inches="tight")
        fig.savefig(out_pdf, dpi=600, bbox_inches="tight")
        plt.close(fig)
        return out_png

    def _save_report(self, df: pd.DataFrame, chart_path: Path) -> Path:
        df_ranked = df.sort_values("f1_score", ascending=False).reset_index(drop=True)
        best = df_ranked.iloc[0].to_dict()

        table_lines = []
        for _, r in df_ranked.iterrows():
            fold_text = "-" if int(r["fold"]) == 0 else str(int(r["fold"]))
            table_lines.append(
                f"| {r['pipeline']} | {r['model']} | {fold_text} | {r['accuracy']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1_score']:.4f} |"
            )

        report = f"""# Unified Best-Fold Comparison Report

## Scope
- Compare best fold of each model across Machine Learning, Deep Learning, and Transformer pipelines.
- Include LLM zero-shot/few-shot results when available.

## Top Result
- **Best overall entry**: {best['pipeline']} / {best['model']}
- **F1-score**: {best['f1_score']:.4f}

## Comparison Table

| Pipeline | Model | Best Fold | Accuracy | Precision | Recall | F1-Score |
|----------|-------|-----------|----------|-----------|--------|----------|
{os.linesep.join(table_lines)}

## Figure
- [cross_pipeline_best_fold_f1_comparison](../figures/cross_pipeline_best_fold_f1_comparison.png)
"""

        out_path = self.output_reports / "cross_pipeline_best_fold_report.md"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report)
        return out_path

    def run(self) -> int:
        rows = self._collect_best_fold_rows()
        if not rows:
            logger.error("No best-fold artifacts found. Run model pipelines first.")
            return 1

        df = pd.DataFrame(rows)
        df.to_csv(self.output_artifacts / "cross_pipeline_best_fold_metrics.csv", index=False, encoding="utf-8")
        chart_path = self._save_plot(df)
        report_path = self._save_report(df, chart_path)

        logger.info(f"Unified comparison CSV: {self.output_artifacts / 'cross_pipeline_best_fold_metrics.csv'}")
        logger.info(f"Unified comparison chart: {chart_path}")
        logger.info(f"Unified comparison report: {report_path}")
        return 0


def main() -> int:
    reporter = CrossPipelineBestFoldReporter()
    return reporter.run()


if __name__ == "__main__":
    raise SystemExit(main())
