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


def parse_overrides(override_args: list) -> dict:
    """Parse ``--set key.path=value`` style overrides into a nested dict.

    Example::

        ["training.lr=5e-4", "training.batch_size=16"]

    Args:
        override_args: List of strings of the form ``"dotted.key=value"``.

    Returns:
        Nested dict of overrides.
    """
    overrides = {}
    for arg in override_args or []:
        if "=" not in arg:
            raise ValueError(
                f"Override must be 'key.path=value', got: {arg}"
            )
        keys_path, value = arg.split("=", 1)
        keys = keys_path.split(".")
        # Try to parse value as int/float/bool, fall back to string
        # yaml.safe_load doesn't handle scientific notation like 5e-4,
        # so try float manually first.
        parsed: object = value
        if value.lower() in ("true", "false"):
            parsed = value.lower() == "true"
        elif value.lower() in ("null", "none"):
            parsed = None
        else:
            try:
                parsed = int(value)
            except ValueError:
                try:
                    parsed = float(value)
                except ValueError:
                    try:
                        parsed = yaml.safe_load(value)
                    except yaml.YAMLError:
                        parsed = str(value)
        cur = overrides
        for k in keys[:-1]:
            cur = cur.setdefault(k, {})
        cur[keys[-1]] = parsed
    return overrides


def apply_overrides(config: dict, overrides: dict) -> dict:
    """Recursively merge ``overrides`` into ``config`` and return a new dict.

    Args:
        config: Base config dict.
        overrides: Nested dict from :func:`parse_overrides`.

    Returns:
        New merged config dict.
    """
    return _deep_merge(config, overrides)
