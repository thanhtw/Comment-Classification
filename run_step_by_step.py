import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


STEP_ORDER = ["machine-learning", "deep-learning", "transformer"]
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
    }


def run_step(name: str, script_path: Path) -> int:
    """Run one pipeline script and return its exit code."""
    _print_section_header(name, script_path)

    if not script_path.exists():
        print(f"[ERROR] Script not found: {script_path}")
        return 1

    start = time.perf_counter()
    cmd = [sys.executable, script_path.name]
    result = subprocess.run(cmd, cwd=str(script_path.parent), check=False)
    elapsed = time.perf_counter() - start

    if result.returncode == 0:
        print(f"[OK] {name} finished in {elapsed:.2f}s")
    else:
        print(f"[FAILED] {name} failed with exit code {result.returncode} after {elapsed:.2f}s")

    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run project pipelines step by step: Machine Learning -> Deep Learning -> Transformer"
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running remaining steps even if one step fails.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=["machine-learning", "deep-learning", "transformer"],
        help="Run only selected pipeline step(s).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    all_steps = _build_steps(project_root)

    selected_keys = args.only if args.only else STEP_ORDER
    steps = [all_steps[key] for key in selected_keys]

    overall_start = time.perf_counter()
    failed = []

    for step in steps:
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
