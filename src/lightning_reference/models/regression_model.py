import lightning as L
import numpy as np
import torch
from models.load_models import prepare_model
from torch import optim
from torchmetrics import Accuracy, ConfusionMatrix, F1Score


class RegressionModel(L.LightningModule):
    def __init__(self, num_classes: int, lr: float, model_name: str):
        super().__init__()

        self.lr = lr

        self.mean_absolute_error = np.inf
        self.minimum_loss = np.inf
        self.model = prepare_model(model_name, num_classes)

        self.save_hyperparameters()

        self.loss_fn = torch.nn.MSELoss()
        self.loss_mea = torch.nn.L1Loss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.float(), y.float()
        logits = self.forward(x)
        loss = self.loss_fn(logits.squeeze(), y)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.float(), y.float()
        logits = self(x)
        loss = self.loss_fn(logits.squeeze(), y)

        loss_mea = self.loss_mea(logits.squeeze(), y)

        if self.minimum_loss > loss:
            self.minimum_loss = loss

        self.log("loss_mea", loss_mea, on_step=False, on_epoch=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("hp_metric", float(self.minimum_loss), on_step=False, on_epoch=True)

    def configure_optimizers(self):
        torch.set_float32_matmul_precision("medium")
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            betas=[0.9, 0.999],
            weight_decay=0.01,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20
        )
        return [optimizer], [scheduler]
