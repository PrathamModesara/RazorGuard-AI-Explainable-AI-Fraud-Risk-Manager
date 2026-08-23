from pathlib import Path
import os

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config() -> dict:
    """Load project configuration from YAML."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def get_project_root() -> Path:
    """Return the project root directory."""
    return PROJECT_ROOT


def get_environment() -> str:
    """Return the current application environment."""
    return os.getenv("APP_ENV", "development")