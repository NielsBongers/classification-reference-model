import numpy as np 
from pathlib import Path 

import torch 
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms 

import albumentations as A 
from albumentations.pytorch import ToTensorV2

from logging_setup import get_logger


class Transforms:
    """Required for Albumentations; has to call this class to apply the list of transforms to each image. 
    """
    def __init__(self, transforms: A.Compose):
        self.transforms = transforms

    def __call__(self, img):
        return self.transforms(image=np.array(img))['image']


def get_transforms() -> dict: 
    """Creates a list of transforms for train and test. 

    Returns:
        dict: transforms created. 
    """
    transform_train = A.Compose(
        [
            A.Resize(299, 299), 
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5), 
            # A.RandomBrightnessContrast(p=0.5),
            A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ToTensorV2()
        ]
    )

    transform_test = A.Compose(
        [
            A.Resize(224, 224), 
            A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ToTensorV2()
        ]
    )
    
    transform_dict = {
        "train": transform_train, 
        "test": transform_test
    }
    
    return transform_dict


def create_image_folder(root: str, split_type: str, batch_size) -> torch.utils.data.DataLoader: 
    """Creates an ImageLoader dataloader from the directory it is pointed at. 

    Args:
        root (str): path to the folder. 
        split_type (str): train or test. 
        batch_size (_type_): batch size for the eventual training. 

    Returns:
        torch.utils.data.DataLoader: dataloader containing the images. 
    """
    logger = get_logger(__name__)
    logger.info(f"Creating image folder based on {root}.")

    transform_dict = get_transforms() 
    dataset = ImageFolder(Path(root, split_type), transform=Transforms(transforms=transform_dict[split_type])) 
    
    logger.info(f"For {split_type}, we have {dataset.class_to_idx}")
    
    data_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=(split_type=="train"))
    
    return data_loader
    

if __name__ == "__main__": 
    root = Path("../Datasets/Temnos demo/X-ray/Sorted") 
    dataset = create_image_folder(root, "train", batch_size=10) 
    
    for item in dataset: 
        print(item) 