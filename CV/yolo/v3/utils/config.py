import os
import yaml


def check_paths(*paths):
    for path in paths:
        if path is None:
            continue
        if not os.path.exists(path):
            if os.path.isdir(path):
                os.makedirs(path)
            else:
                raise ValueError(f"File {path} does not exist")


def get_config(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config["model"], config["data"], config["train"]