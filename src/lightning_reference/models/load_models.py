import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from utils.logging_setup import get_logger


def prepare_model(model_name: str, num_classes: int) -> None:
    """Prepares and returns pre-created models.

    Args:
        model_name (str): Model name.
        num_classes (int): Number of classes to use for the model.

    Returns:
        _type_: Specified model with ImageNet weights.
    """
    logger = get_logger(__name__)

    logger.info(f"Creating model {model_name} with {num_classes} classes.")

    if model_name == "efficientnet":
        model = models.efficientnet_v2_s(
            weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1,
            classes=num_classes,
        )
        model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)

    if model_name == "convnext":
        model = models.convnext_small(
            weights=torchvision.models.ConvNeXt_Small_Weights.IMAGENET1K_V1
        )
        model.classifier[2] = nn.Sequential(nn.Linear(768, num_classes))

    if model_name == "inception":
        model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        model.aux_logits = False
        model.fc = nn.Linear(2048, num_classes)

    if model_name == "resnet152":
        model = models.resnet152(weights=torchvision.models.ResNet152_Weights.DEFAULT)
        fc_layer_output = model.fc.in_features
        model.fc = torch.nn.Linear(fc_layer_output, num_classes)

    if model_name == "resnet50":
        model = models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        fc_layer_output = model.fc.in_features
        model.fc = torch.nn.Linear(fc_layer_output, num_classes)

    if model_name == "resnet18":
        model = models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        fc_layer_output = model.fc.in_features
        model.fc = torch.nn.Linear(fc_layer_output, num_classes)

    return model
