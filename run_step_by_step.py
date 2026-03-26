import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_step(name: str, script_path: Path) -> int:
    """Run one pipeline script and return its exit code."""
    print("\n" + "=" * 80)
    print(f"STEP: {name}")
    print(f"SCRIPT: {script_path}")
    print("=" * 80)

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
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    steps = [
        ("Machine Learning", project_root / "Machine_Learning_10fold" / "machine-learning.py"),
        ("Deep Learning", project_root / "DeepLearning_10fold" / "LSTMvsBiLSTM.py"),
        ("Transformer", project_root / "Transformer_10fold" / "BertvsRoberta.py"),
    ]

    overall_start = time.perf_counter()
    failed = []

    for name, script in steps:
        code = run_step(name, script)
        if code != 0:
            failed.append((name, code))
            if not args.continue_on_error:
                print("\nStopping because a step failed. Use --continue-on-error to keep going.")
                break

    total_elapsed = time.perf_counter() - overall_start
    print("\n" + "-" * 80)
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
