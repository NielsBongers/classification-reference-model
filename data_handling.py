import numpy as np 
import shutil 
from PIL import Image as Im 
from pathlib import Path 
from sklearn.model_selection import train_test_split

import torch 
from torch.utils.data import Dataset 
from torchvision.datasets import ImageFolder 
import albumentations as A 
from albumentations.pytorch import ToTensorV2

from logging_setup import get_logger 

def create_split_names(root: str, test_size=0.2): 
    file_path_list = list(Path(root, "raw").glob("**/*")) 
    
    sorted_classes_lists = {
        "present": [], 
        "removed": [] 
    }
    
    for file_path in file_path_list: 
        uuid, increment, name = file_path.stem.split(" - ") 
        
        if increment.isalpha(): 
            sorted_classes_lists["removed"].append(file_path)
        else: 
            sorted_classes_lists["present"].append(file_path)

    destination_folder_name = "Sorted" 
    Path(root, destination_folder_name).mkdir(exist_ok=True) 

    for sorting_class in sorted_classes_lists.keys(): 
        train_list, test_list = train_test_split(sorted_classes_lists[sorting_class], test_size=test_size, random_state=42) 
        
        train_test_dict = {
            "train": train_list, 
            "test": test_list
        }
        
        for train_test_key in train_test_dict.keys(): 
            Path(root, destination_folder_name, train_test_key).mkdir(exist_ok=True) 
            for file_path in train_test_dict[train_test_key]: 
                img = Im.open(file_path) 
                width, height = img.size 
                img = img.crop((0, 450, width-300, height-250))
                img.save(Path(root, destination_folder_name, train_test_key, file_path.name))

if __name__ == "__main__": 
    root = "../Datasets/Temnos demo/X-ray" 
    create_split_names(root) 