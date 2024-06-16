import random
from pathlib import Path
from typing import List

import lightning as L
import pandas as pd
import torch
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from PIL import Image as Im
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_list: List[Path], transform: v2.Compose):
        self.image_list = image_list
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = self.image_list[idx]

        img = Im.open(image_path)

        rotation_options = [0, 90, 180, 270]

        selected_index = random.randint(0, len(rotation_options) - 1)
        selected_rotation = rotation_options[selected_index]
        img = img.rotate(selected_rotation)

        img = self.transform(img)

        return img, selected_index


if __name__ == "__main__":
    image_path_list = list(Path(r"F:\COCO\train2017").glob("**/*"))

    transform = v2.Compose(
        [
            # v2.RandomResizedCrop(size=(224, 224), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    regression_dataset = CustomDataset(image_list=image_path_list, transform=transform)

    img, selected_rotation = regression_dataset[0]

    print(selected_rotation)

    import matplotlib.pyplot as plt

    plt.imshow(img.T)
    plt.show()
