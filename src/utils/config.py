"""Environment and configuration management.

Loads configuration from .env file and provides centralized access.
"""

import os
from pathlib import Path
from typing import Dict, Optional


def get_project_root() -> Path:
    """Resolve project root from environment or repository layout.
    
    Returns:
        Path: Project root directory
    """
    env_root = os.getenv("PROJECT_ROOT", "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    # config.py is at <repo>/src/utils/config.py; repository root is parents[2].
    return Path(__file__).resolve().parents[2]


def load_env_file() -> None:
    """Load simple KEY=VALUE pairs from project .env into process environment.
    
    Skips comments and invalid lines. Does not override existing env variables.
    """
    env_path = get_project_root() / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def get_config(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get configuration value from environment.
    
    Args:
        key: Configuration key
        default: Default value if key not found
        
    Returns:
        Configuration value or default
    """
    load_env_file()
    return os.getenv(key, default)


def get_data_file() -> Path:
    """Get data file path from configuration.
    
    Returns:
        Path: Data file path (default: ./data/Dataset.csv)
    """
    load_env_file()
    data_file = os.getenv("DATA_FILE", "./data/Dataset.csv")
    return Path(data_file)


def get_results_root() -> Path:
    """Get results root directory from configuration.
    
    Returns:
        Path: Results root (default: ./results)
    """
    load_env_file()
    results_root = os.getenv("RESULTS_ROOT", "./results")
    return Path(results_root)


def get_groq_api_key() -> Optional[str]:
    """Get Groq API key from environment.
    
    Supports both Groq_API_KEY and GROQ_API_KEY names.
    
    Returns:
        API key or None if not configured
    """
    load_env_file()
    return os.getenv("GROQ_API_KEY") or os.getenv("Groq_API_KEY")


def get_llm_model_names() -> list[str]:
    """Get LLM model names from configuration.
    
    Returns:
        List of model IDs (default: ["openai/gpt-oss-20b", "llama-3.3-70b-versatile"])
    """
    load_env_file()
    env_models = os.getenv("LLM_MODEL_NAMES", "").strip()
    if env_models:
        return [m.strip() for m in env_models.split(",") if m.strip()]
    return ["openai/gpt-oss-20b", "llama-3.3-70b-versatile"]


def get_llm_max_samples() -> int:
    """Get max samples for LLM evaluation.
    
    Returns:
        Max samples (default: 200)
    """
    load_env_file()
    try:
        return int(os.getenv("LLM_MAX_SAMPLES", "200"))
    except ValueError:
        return 200


class Config:
    """Singleton-like configuration manager."""
    
    _instance: Optional["Config"] = None
    
    def __new__(cls) -> "Config":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self) -> None:
        if self._initialized:
            return
        load_env_file()
        self._initialized = True
    
    @property
    def project_root(self) -> Path:
        """Project root directory."""
        return get_project_root()
    
    @property
    def data_file(self) -> Path:
        """Data file path."""
        return get_data_file()
    
    @property
    def results_root(self) -> Path:
        """Results root directory."""
        return get_results_root()
    
    @property
    def groq_api_key(self) -> Optional[str]:
        """Groq API key."""
        return get_groq_api_key()
    
    @property
    def llm_model_names(self) -> list[str]:
        """LLM model names."""
        return get_llm_model_names()
    
    @property
    def llm_max_samples(self) -> int:
        """Max samples for LLM."""
        return get_llm_max_samples()
