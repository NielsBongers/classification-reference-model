import yaml 


def parse_yaml(file_name: str) -> dict: 
    """Loads YAML data from a specified file and returns a dict. 

    Args:
        file_name (str): YAML hyperparameter file. 

    Returns:
        dict: contents of the file
    """
    with open("Parameters/default.yaml") as f: 
        yaml_data = yaml.safe_load(f)
    return yaml_data

