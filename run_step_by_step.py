import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

STEP_ORDER = ["machine-learning", "deep-learning", "transformer", "llm-groq", "cross-comparison"]
SECTION_WIDTH = 80


class CrossPipelineBestFoldReporter:
    """Generate unified best-fold comparison reports across all pipelines."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
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


STEP_ORDER = ["machine-learning", "deep-learning", "transformer", "llm-groq", "cross-comparison"]
SECTION_WIDTH = 80


@dataclass(frozen=True)
class PipelineStep:
    name: str
    script_path: Path


def _print_section_header(name: str, script_path: Path) -> None:
    print("\n" + "=" * SECTION_WIDTH)
    print(f"STEP: {name}")
    print(f"SCRIPT: {script_path}")
    print("=" * SECTION_WIDTH)


def _build_steps(project_root: Path) -> dict[str, PipelineStep]:
    return {
        "machine-learning": PipelineStep(
            name="Machine Learning",
            script_path=project_root / "src" / "models" / "machine_learning_pipeline.py",
        ),
        "deep-learning": PipelineStep(
            name="Deep Learning",
            script_path=project_root / "src" / "models" / "deep_learning_pipeline.py",
        ),
        "transformer": PipelineStep(
            name="Transformer",
            script_path=project_root / "src" / "models" / "transformer_pipeline.py",
        ),
        "llm-groq": PipelineStep(
            name="LLM Groq Inference",
            script_path=project_root / "src" / "models" / "llm_groq_inference.py",
        ),
        "cross-comparison": PipelineStep(
            name="Cross-Pipeline Best-Fold Comparison",
            script_path=project_root / "src" / "models" / "cross_pipeline_best_fold_report.py",
        ),
    }


def run_step(name: str, script_path: Path) -> int:
    """Run one pipeline script and return its exit code."""
    _print_section_header(name, script_path)

    if not script_path.exists():
        print(f"[ERROR] Script not found: {script_path}")
        return 1

    start = time.perf_counter()
    cmd = [sys.executable, str(script_path)]
    run_env = os.environ.copy()
    run_env["PYTHONNOUSERSITE"] = "1"
    result = subprocess.run(
        cmd,
        cwd=str(Path(__file__).resolve().parent),
        check=False,
        env=run_env,
    )
    elapsed = time.perf_counter() - start

    if result.returncode == 0:
        print(f"[OK] {name} finished in {elapsed:.2f}s")
    else:
        print(f"[FAILED] {name} failed with exit code {result.returncode} after {elapsed:.2f}s")

    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run project pipelines step by step and generate unified best-fold comparison report"
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running remaining steps even if one step fails.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=["machine-learning", "deep-learning", "transformer", "llm-groq", "cross-comparison"],
        help="Run only selected pipeline step(s).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    all_steps = _build_steps(project_root)

    selected_keys = args.only if args.only else STEP_ORDER
    steps = [all_steps[key] for key in selected_keys]

    overall_start = time.perf_counter()
    failed = []

    for key, step in zip(selected_keys, steps):
        if key == "cross-comparison":
            # Run cross-pipeline reporting directly
            print("\n" + "=" * SECTION_WIDTH)
            print(f"STEP: {step.name}")
            print("=" * SECTION_WIDTH)
            start = time.perf_counter()
            reporter = CrossPipelineBestFoldReporter(project_root)
            code = reporter.run()
            elapsed = time.perf_counter() - start
            if code == 0:
                print(f"[OK] {step.name} finished in {elapsed:.2f}s")
            else:
                print(f"[FAILED] {step.name} failed with exit code {code} after {elapsed:.2f}s")
        else:
            code = run_step(step.name, step.script_path)
        
        if code != 0:
            failed.append((step.name, code))
            if not args.continue_on_error:
                print("\nStopping because a step failed. Use --continue-on-error to keep going.")
                break

    total_elapsed = time.perf_counter() - overall_start
    print("\n" + "-" * SECTION_WIDTH)
    print(f"Total elapsed time: {total_elapsed:.2f}s")

    if failed:
        print("Run finished with failures:")
        for name, code in failed:
            print(f"- {name}: exit code {code}")
        return 1

    print("All steps completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
