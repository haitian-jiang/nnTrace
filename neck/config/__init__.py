from pathlib import Path
from omegaconf import OmegaConf

def load_config():
    current_dir = Path(__file__).resolve().parent
    config_path_file = current_dir / "placeholder.yaml"
    with open(config_path_file) as f:
        d = OmegaConf.load(f)
    config_path = d.config_path
    if not Path(config_path).is_absolute():
        config_path = current_dir / config_path
    with open(config_path) as config_file:
        config = OmegaConf.load(config_file)
    return config


CONFIG = load_config()
