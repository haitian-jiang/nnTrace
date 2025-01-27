import os
from pathlib import Path
from omegaconf import OmegaConf

def load_config():
    config_path = None
    env_config = os.environ.get("NECK_CONFIG_PATH", None)
    if env_config is None:  # Load config_path from ./placeholder.yaml
        current_dir = Path(__file__).resolve().parent
        placeholder_path = current_dir / "placeholder.yaml"
        with open(placeholder_path) as f:
            d = OmegaConf.load(f)
        config_path = d.config_path
    else:
        config_path = env_config
    if not Path(config_path).is_absolute():
        config_path = current_dir / config_path
    with open(config_path) as config_file:
        config = OmegaConf.load(config_file)
    return config


CONFIG = load_config()
