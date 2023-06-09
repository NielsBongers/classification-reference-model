import yaml 
from pathlib import Path 


def parse_yaml(file_name: str) -> dict: 
    """Loads YAML data from a specified file and returns a dict. 

    Args:
        file_name (str): YAML hyperparameter file. 

    Returns:
        dict: contents of the file
    """
    with open(Path("parameters", file_name)) as f: 
        yaml_data = yaml.safe_load(f)
    return yaml_data


def save_yaml(file_path: str, yaml_data: dict): 
    with open(file_path, 'w') as f: 
        yaml.dump(yaml_data, f, default_flow_style=False) 
