# File: src/data/config_loader.py
"""
Single place to load params.yaml and pipeline_config.yaml.
All other src/ files import from here — no hardcoded values anywhere.
"""
import os
import yaml
import logging

logger = logging.getLogger(__name__)

# These paths work both on host and inside container
_HOST_PARAMS = "params.yaml"
_HOST_CONFIG = "config/pipeline_config.yaml"

_CONTAINER_PARAMS = "/opt/airflow/params.yaml"
_CONTAINER_CONFIG = "/opt/airflow/config/pipeline_config.yaml"


def _find_file(host_path: str, container_path: str) -> str:
    if os.path.exists(container_path):
        return container_path
    if os.path.exists(host_path):
        return host_path
    raise FileNotFoundError(
        f"Config file not found at '{host_path}' or '{container_path}'"
    )


def load_params() -> dict:
    path = _find_file(_HOST_PARAMS, _CONTAINER_PARAMS)
    with open(path) as f:
        params = yaml.safe_load(f)
    logger.info(f"Loaded params from: {path}")
    return params


def load_pipeline_config() -> dict:
    path = _find_file(_HOST_CONFIG, _CONTAINER_CONFIG)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Loaded pipeline config from: {path}")
    return cfg


def get(section: str, key: str, fallback=None):
    """Convenience: load params and get params[section][key]."""
    try:
        return load_params()[section][key]
    except KeyError:
        if fallback is not None:
            return fallback
        raise