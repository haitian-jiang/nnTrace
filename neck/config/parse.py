import yaml

def get_config_file_path():
    with open("./config.yaml", "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)
    