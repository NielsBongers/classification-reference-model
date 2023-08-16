import yaml
from pathlib import Path

from logging_setup import get_logger


def parse_yaml(file_name="default.yaml") -> dict:
    """Loads YAML data from a specified file and returns a dict.

    Args:
        file_name (str): YAML hyperparameter file.

    Returns:
        dict: contents of the file
    """
    logger = get_logger(__name__)
    logger.info(f"Loading {file_name} for hyperparameter data.")

    with open(Path("parameters", file_name)) as f:
        yaml_data = yaml.safe_load(f)

    logger.info(f"YAML data:\n{yaml.dump(yaml_data)}")

    return yaml_data


def save_yaml(file_path: str, yaml_data: dict):
    with open(file_path, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False)
