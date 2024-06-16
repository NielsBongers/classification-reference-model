from pathlib import Path

import lightning as L
import pandas as pd
import torch
from data_handling.custom_dataset import CustomDataset
from data_handling.data_module import CustomDataModule
from models.classification_model import ClassificationModel

torch.set_float32_matmul_precision("medium")

if __name__ == "__main__":
    checkpoint_path = Path(
        r"C:\Users\Niels\Desktop\lightning-reference-model\logs\Statice\version_27\checkpoints\best-checkpoint.ckpt"
    )

    model = ClassificationModel.load_from_checkpoint(checkpoint_path=checkpoint_path)

    data_module = CustomDataModule("")

    df = pd.read_csv(
        r"C:\Users\Niels\Desktop\python-sandbox\statice_table\results\data_output.csv"
    )

    dataset = CustomDataset(df=df, images_path="", transform=data_module.transform)

    import matplotlib.pyplot as plt

    model = model.eval()

    uuids = []
    inference = []
    actual = []

    for img, label, uuid in dataset:
        with torch.no_grad():  # Disable gradient calculation
            res = model(img.unsqueeze(0).to("cuda"))
            probabilities = torch.sigmoid(res)
            predicted_class = (probabilities >= 0.5).float().detach().cpu().item()

            uuids.append(uuid)
            inference.append(predicted_class)
            actual.append(label.item())

            print(f"True label: {label.item()}. Model prediction: {predicted_class}")

    result_df = pd.DataFrame(
        {
            "UUID": uuids,
            "torch": inference,
            "label": actual,
        }
    )

    result_df.to_csv("results/pytorch_results.csv")
