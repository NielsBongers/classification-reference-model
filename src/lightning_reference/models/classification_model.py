import lightning as L
import torch
from models.load_models import prepare_model
from torch import optim
from torchmetrics import Accuracy, ConfusionMatrix, F1Score


class ClassificationModel(L.LightningModule):
    def __init__(self, num_classes: int, lr: float, model_name: str):
        super().__init__()

        self.NUM_CLASSES = num_classes
        self.lr = lr

        self.best_acc = -1

        self.model = prepare_model(model_name, num_classes)

        self.save_hyperparameters()

        self.train_accuracy = Accuracy(task="multiclass", num_classes=self.NUM_CLASSES)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.NUM_CLASSES)
        self.val_f1 = F1Score(task="multiclass", num_classes=self.NUM_CLASSES)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        acc = self.train_accuracy(logits, y)
        self.log("train_accuracy", acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.val_accuracy(logits, y)
        f1 = self.val_f1(logits, y)

        if self.best_acc > acc:
            self.best_acc = acc

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_accuracy", acc, on_step=False, on_epoch=True)
        self.log("val_f1", f1, on_step=False, on_epoch=True)
        self.log("hp_metric", self.best_acc, on_step=False, on_epoch=True)

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
