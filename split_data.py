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
    """Creates a split based on the image's names, sorting into train and test, based on a Raw folder in the specified root.

    Args:
        root (str): folder from which to start.
        test_size (float, optional): Split between train and test. Defaults to 0.2.
    """
    logger = get_logger(__name__)

    logger.info(f"Creating split in {str(root)}.")
    file_path_list = list(Path(root, "raw").glob("**/*"))

    sorted_classes_lists = {"present": [], "removed": []}

    for file_path in file_path_list:
        uuid, increment, name = file_path.stem.split(" - ")

        if increment.isalpha():
            sorted_classes_lists["removed"].append(file_path)
        else:
            sorted_classes_lists["present"].append(file_path)

    destination_folder_name = "Sorted"
    Path(root, destination_folder_name).mkdir(exist_ok=True)

    split_list = ["train", "test"]

    for subfolder in split_list:
        Path(root, destination_folder_name, subfolder).mkdir(exist_ok=True)
        for sorted_class in sorted_classes_lists.keys():
            Path(root, destination_folder_name, subfolder, sorted_class).mkdir(
                exist_ok=True
            )

    for sorted_class in sorted_classes_lists.keys():
        train_list, test_list = train_test_split(
            sorted_classes_lists[sorted_class], test_size=test_size, random_state=42
        )

        split_dict = {"train": train_list, "test": test_list}

        for split_type in split_dict.keys():
            logger.info(f"Splitting in {sorted_class} and split type {split_type}.")
            for image_path in split_dict[split_type]:
                img = Im.open(image_path)
                width, height = img.size
                img = img.crop((0, 450, width - 300, height - 250))
                img.save(
                    Path(
                        root,
                        destination_folder_name,
                        split_type,
                        sorted_class,
                        image_path.name,
                    )
                )
    logger.info("Completed splitting.")


def create_split_folders(root: str, test_size=0.2):
    logger = get_logger(__name__)

    file_path_list = list(Path(root, "raw").glob("**/*"))

    alkaline_dict = {}
    lithium_dict = {}

    data_classes = {"alkaline": alkaline_dict, "lithium": lithium_dict}

    for file_path in file_path_list:
        if not file_path.is_file():
            continue

        class_key = file_path.parent.name
        image_id, image_count = file_path.stem.split(" ")

        if image_id not in data_classes[class_key]:
            data_classes[class_key][image_id] = []

        data_classes[class_key][image_id].append(file_path)

    for class_key in data_classes.keys():
        print(class_key)

        training_keys, test_keys = train_test_split(
            list(data_classes[class_key].keys()), test_size=test_size
        )

        for training_key in training_keys:
            print(f"Class key: {class_key}, training key: {training_key}")
            print(data_classes[class_key])

            image_source_paths = data_classes[class_key][training_key]

            for image_source_path in image_source_paths:
                image_destination_path = Path(root, "sorted", "train", class_key)
                image_destination_path.mkdir(parents=True, exist_ok=True)

                shutil.copy(
                    image_source_path,
                    str(image_destination_path / Path(image_source_path.name)),
                )

        for test_key in test_keys:
            image_source_paths = data_classes[class_key][test_key]

            for image_source_path in image_source_paths:
                image_destination_path = Path(root, "sorted", "test", class_key)
                image_destination_path.mkdir(parents=True, exist_ok=True)

                shutil.copy(
                    image_source_path,
                    str(image_destination_path / Path(image_source_path.name)),
                )


if __name__ == "__main__":
    root = Path(
        "../Datasets/Battery classification/Wisse's preliminary data/Sorted/alkaline_lithium_cropped"
    )
    create_split_folders(root)
