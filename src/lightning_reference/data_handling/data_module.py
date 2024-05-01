from pathlib import Path

import lightning as L
import pandas as pd
import torch
from data_handling.custom_dataset import CustomDataset
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2


class CustomDataModule(L.LightningDataModule):
    """Custom data module based on PyTorch Lightning.

    Args:
        L (_type_): PyTorch Lightning.
    """

    def __init__(self, data_dir: str, batch_size: int = 32, df: pd.DataFrame = None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = v2.Compose(
            [
                v2.RandomResizedCrop(size=(224, 224), antialias=True),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.df = df

    def prepare_data(self) -> None:
        # Nothing here for now. Useful for downloading etc. - single threaded operations.
        pass

    def setup(self, stage: str):
        if self.df is None:
            self.train_dataset = ImageFolder(
                Path(self.data_dir) / "train", transform=self.transform
            )
            self.val_dataset = ImageFolder(
                Path(self.data_dir) / "test", transform=self.transform
            )

        else:
            custom_dataset = CustomDataset(
                df=self.df,
                images_path=self.data_dir,
                transform=self.transform,
            )

            self.train_dataset, self.val_dataset = random_split(
                dataset=custom_dataset,
                lengths=[0.8, 0.2],
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            persistent_workers=True,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            persistent_workers=True,
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            persistent_workers=True,
        )
