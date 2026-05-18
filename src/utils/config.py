"""
YAML config loader and merger.

Loads individual YAML configs and merges them into a single experiment
configuration dictionary.
"""

import os
import yaml
from copy import deepcopy


def load_config(path: str) -> dict:
    """Load a single YAML config file.

    Args:
        path: Absolute or relative path to the YAML file.

    Returns:
        Dictionary of config values.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        yaml.YAMLError: If the YAML is malformed.
    """
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    if config is None:
        config = {}
    return config


def merge_configs(*configs: dict) -> dict:
    """Deep-merge multiple config dicts (later overrides earlier).

    Args:
        *configs: Variable number of config dictionaries.

    Returns:
        Merged config dictionary.
    """
    merged = {}
    for cfg in configs:
        merged = _deep_merge(merged, deepcopy(cfg))
    return merged


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_experiment_config(
    model_config_path: str = None,
    preprocessing_config_path: str = None,
    experiment_config_path: str = None,
) -> dict:
    """Load and merge all config files for an experiment run.

    Args:
        model_config_path: Path to model YAML config.
        preprocessing_config_path: Path to preprocessing YAML config.
        experiment_config_path: Path to experiment YAML config.

    Returns:
        Merged config dict with keys 'model', 'preprocessing', 'experiment'.
    """
    config = {}

    if model_config_path:
        config["model"] = load_config(model_config_path)

    if preprocessing_config_path:
        config["preprocessing"] = load_config(preprocessing_config_path)

    if experiment_config_path:
        experiment = load_config(experiment_config_path)
        # Training params from experiment config go at top level
        config["experiment"] = experiment
        if "training" in experiment:
            config["training"] = experiment["training"]

    return config


def save_config(config: dict, path: str) -> None:
    """Save a config dict as YAML.

    Args:
        config: Config dictionary to save.
        path: Output file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
