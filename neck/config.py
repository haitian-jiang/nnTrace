import os
from pathlib import Path
from omegaconf import OmegaConf


def _load_config():
    config_path = os.environ.get("NECK_CONFIG_PATH", None)

    if config_path is None:
        return None
        # config_path = "/home/ubuntu/repos/NECK/config/gpt.yaml"
        # raise ValueError("Environment variable NECK_CONFIG_PATH is not set.")

    if not Path(config_path).is_absolute():
        config_path = Path.cwd() / config_path

    assert Path(config_path).exists(), f"Config file {config_path} does not exist."
    with open(config_path) as config_file:
        config = OmegaConf.load(config_file)

    return config


CONFIG = _load_config()

__all__ = ["CONFIG"]
