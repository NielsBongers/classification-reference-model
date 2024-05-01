import lightning as L
import pandas as pd
import torch
from data_handling.data_module import CustomDataModule
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from models.classification_model import ClassificationModel
from models.regression_model import RegressionModel
from utils.parse_hyper import parse_yaml

torch.set_float32_matmul_precision("medium")


def main():
    config = parse_yaml("src/lightning_reference/parameters/default.yaml")

    root = config["root"]
    run_name = config["run name"]
    model_name = config["model"]

    num_classes = config["number classes"] if not config["regression"] else 1
    save_models = config["save model"]

    lr = config["learning rate"]
    max_epochs = config["max epochs"]
    batch_size = config["real batch size"]
    gradient_steps = config["accumulation steps"]

    df = pd.read_csv(
        "../Datasets/Temnos image photos/Temnos demo dataset - corrected.csv"
    )

    data_module = CustomDataModule(root, batch_size, df=df)
    model = RegressionModel(lr=lr, num_classes=num_classes, model_name=model_name)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    # early_stopping = EarlyStopping("val_loss", patience=7)

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        logger=TensorBoardLogger(save_dir="logs/", name=run_name),
        callbacks=[lr_monitor],
        accumulate_grad_batches=gradient_steps,
        enable_checkpointing=save_models,
        log_every_n_steps=8,
    )

    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
