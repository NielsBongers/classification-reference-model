from pathlib import Path

import yaml
from utils.logging_setup import get_logger


def parse_yaml(file_name="parameters/default.yaml") -> dict:
    """Loads YAML data from a specified file and returns a dict.

    Args:
        file_name (str): YAML hyperparameter file.

    Returns:
        dict: contents of the file
    """
    logger = get_logger(__name__)
    logger.info(f"Loading {file_name} for hyperparameter data.")

    with open(Path(file_name), encoding="utf8") as f:
        yaml_data = yaml.safe_load(f)

    logger.info(f"YAML data:\n{yaml.dump(yaml_data)}")

    return yaml_data
