from logging_setup import get_logger
from parse_hyper import parse_yaml
from track_runs import create_run_folder
from data_handling import create_image_folder
from model import prepare_model 

def main(): 
    logger = get_logger(__name__) 
    logger.info("Starting run.") 

    parameters = parse_yaml() 
    run_path = create_run_folder(parameters)
    
    root = parameters["root"] 
    # root = "../Datasets/Temnos demo/X-ray/Sorted"
    
    data_loaders = {
        "train": create_image_folder(root, "train", batch_size=8), 
        "test": create_image_folder(root, "test", batch_size=8)
    }
    
    prepare_model(data_loaders, parameters)

if __name__ == "__main__": 
    main() 