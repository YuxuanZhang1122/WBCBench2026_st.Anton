import yaml
from typing import Any, Dict
import os


class Config:
    def __init__(self, config_dict: Dict[str, Any]):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def __repr__(self):
        return str(self.__dict__)


def load_config(config_path: str) -> Config:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    return Config(config_dict)


def save_config(config: Config, save_path: str):
    config_dict = _config_to_dict(config)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def _config_to_dict(config: Config) -> Dict[str, Any]:
    result = {}
    for key, value in config.__dict__.items():
        if isinstance(value, Config):
            result[key] = _config_to_dict(value)
        else:
            result[key] = value
    return result
