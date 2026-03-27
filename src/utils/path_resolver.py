"""Path resolution and directory management utilities."""

from pathlib import Path
from typing import Dict

from .config import get_project_root, get_results_root


def get_pipeline_results_dirs(pipeline_name: str) -> Dict[str, Path]:
    """Get standardized results directories for a pipeline.
    
    Args:
        pipeline_name: Pipeline name (e.g., 'machine_learning', 'deep_learning')
        
    Returns:
        Dict with keys: base, artifacts, figures, models, reports
    """
    project_root = get_project_root()
    results_root = get_results_root()
    if not results_root.is_absolute():
        results_root = (project_root / results_root).resolve()
    base = results_root / pipeline_name
    
    dirs = {
        "base": base,
        "artifacts": base / "artifacts",
        "figures": base / "figures",
        "models": base / "models",
        "reports": base / "reports",
    }
    
    # Create all directories
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    
    return dirs


def get_llm_results_dir() -> Path:
    """Get standardized LLM results directory.
    
    Returns:
        Path to results/llm/
    """
    project_root = get_project_root()
    results_root = get_results_root()
    if not results_root.is_absolute():
        results_root = (project_root / results_root).resolve()
    out_dir = results_root / "llm"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def get_cross_pipeline_dirs() -> Dict[str, Path]:
    """Get standardized cross-pipeline comparison directories.
    
    Returns:
        Dict with keys: artifacts, figures, reports
    """
    project_root = get_project_root()
    results_root = get_results_root()
    if not results_root.is_absolute():
        results_root = (project_root / results_root).resolve()
    
    dirs = {
        "artifacts": results_root / "artifacts",
        "figures": results_root / "figures",
        "reports": results_root / "reports",
    }
    
    # Create all directories
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    
    return dirs
