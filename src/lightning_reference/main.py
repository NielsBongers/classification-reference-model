from pathlib import Path

import lightning as L
import pandas as pd
import torch
from data_handling.data_module import CustomDataModule
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
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

    image_list = list(Path(r"F:\COCO\train2017").glob("**/*"))[0:100000]

    data_module = CustomDataModule(image_list=image_list, batch_size=batch_size)
    # model = ClassificationModel(lr=lr, num_classes=num_classes, model_name=model_name)

    checkpoint_path = r"D:\Desktop\classification-reference-model\logs\Rotation classification\version_16\checkpoints\best-checkpoint.ckpt"

    model = ClassificationModel.load_from_checkpoint(checkpoint_path=checkpoint_path)

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",  # Metric to monitor
        mode="min",  # Mode to determine the best model ('min' or 'max')
        save_top_k=1,  # Save only the best model
        filename="best-checkpoint",  # Filename for the best checkpoint
    )

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        logger=TensorBoardLogger(save_dir="logs/", name=run_name),
        callbacks=[lr_monitor, checkpoint_callback],
        accumulate_grad_batches=gradient_steps,
        enable_checkpointing=save_models,
        log_every_n_steps=8,
    )

    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=checkpoint_path,
    )


if __name__ == "__main__":
    main()
