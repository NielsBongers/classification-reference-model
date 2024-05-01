from pathlib import Path

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
    def __init__(self, df: pd.DataFrame, images_path: str, transform: v2.Compose):
        self.df = df
        self.images_path = images_path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_name = row["UUID"] + " - " + row["phone_name"] + " - weight.png"

        img = Im.open(Path(self.images_path, image_name))

        img = self.transform(img)
        weight = row["weight"]

        return img, weight


if __name__ == "__main__":
    df = pd.read_csv(
        "../Datasets/Temnos image photos/Temnos demo dataset - corrected.csv"
    )

    images_path = "../Datasets/Temnos image photos/RGB/raw"

    transform = v2.Compose(
        [
            # v2.RandomResizedCrop(size=(224, 224), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    regression_dataset = CustomDataset(
        df=df, images_path=images_path, transform=transform
    )

    img, weight = regression_dataset[0]

    print(weight)

    import matplotlib.pyplot as plt

    plt.imshow(img.T)
    plt.show()
