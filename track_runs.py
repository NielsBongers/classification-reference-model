import re 
from pathlib import Path 
from parse_hyper import parse_yaml, save_yaml


def create_run_folder(parameters: dict): 
    """Checks the hyperparameter YAML for the run name, either creates orincrements the existing folders. 

    Args:
        parameters (dict): values for the hyperparameters. 
    """
    run_name = parameters["run name"] 

    folder_increment_list = [] 
    for path in Path("runs").glob(f"**/{run_name}*"): 
        try: 
            folder_increment = int(re.findall(" ([0-9]+)$", path.name)[0]) 
        except: 
            folder_increment = 0 
        folder_increment_list.append(folder_increment)
        
    folder_increment_list.sort()

    if folder_increment_list: 
        next_increment = folder_increment_list[-1] + 1 
    else: 
        next_increment = 1 

    run_path = Path("runs", run_name + " " + str(next_increment)) 

    Path(run_path).mkdir(parents=True) 
    Path(run_path, "parameters").mkdir(parents=True) 
    Path(run_path, "models").mkdir(parents=True) 
    
    save_yaml(Path(run_path, "parameters", "hyperparameters.yaml"), parameters)
    
if __name__ == "__main__": 
    parameters = parse_yaml("default.yaml")
    create_run_folder(parameters) 