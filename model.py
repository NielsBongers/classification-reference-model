from pathlib import Path 

import torch 
import torch.nn as nn 
import torch.optim as optim 
import torchvision 
import torchvision.models as models 
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MulticlassConfusionMatrix  

from data_handling import Transforms, get_transforms
from logging_setup import get_logger

def save_model(model, epoch: int, parameters: dict, run_path) -> None: 
    """Save a TorchVision model to a .pt file, with the date/time specified.

    Args:
        model: trained model. 
        epoch: current epoch. 
        parameters: dict with hyperparameters and names. 
    """
    
    if parameters["only save best"]: 
        torch.save(model, Path(run_path, "models", "best.pt")) 
    else: 
        torch.save(model, Path(run_path, "models", "model saved at " + str(epoch) + ".pt")) 


def load_model(model_path: str): 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device) 
    return model 
    
    
def evaluate_model(model, img): 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    transforms = Transforms(transforms=get_transforms()["test"]) 
    img = transforms(img).unsqueeze(0).to(device) 
    
    model.eval() 
    
    outputs = model(img) 
    confidence, _ = torch.max(torch.nn.functional.softmax(outputs, dim=1), 1)
    confidence = confidence.item() 
    
    _, preds = torch.max(outputs, 1) 
    prediction = preds.detach().cpu().item()
    
    return prediction, confidence

    
def train_model(model, data_loader, device, loss_function, optimizer, scheduler, parameters): 
    logger = get_logger(__name__)
    logger.info("Starting model training.")
    
    writer = SummaryWriter(log_dir=Path(parameters["run path"], "tensorboard")) 
    
    gradient_accumulation_steps = round(parameters["desired batch size"]/parameters["real batch size"]) 
    logger.info(f"Using {gradient_accumulation_steps} gradient accumulation steps to achieve a batch size of {parameters['desired batch size']} with batches of {parameters['real batch size']} items.")
    
    best_accuracy = 0.0 

    for epoch in range(parameters["epochs"]):
        logger.info(f'Epoch {epoch}/{parameters["epochs"] - 1}')
        
        train_accuracy = 0 
        test_accuracy = 0 

        for phase in ["train", "test"]:
            logger.info(f"\tCurrently in {phase}.") 
            
            if phase == "train":
                model.train() 
            else:
                model.eval() 

            total_count = 0
            running_corrects = 0
            
            metric = MulticlassConfusionMatrix(num_classes=2).to(device) 

            optimizer.zero_grad() 

            for index, (inputs, labels) in enumerate(data_loader[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                if index % 10 == 0: 
                    logger.info(f"\t\tAt {index} batches processed")

                with torch.set_grad_enabled(phase=="train"):
                    
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    
                    if phase == "train": 
                        loss = loss_function(outputs, labels) / gradient_accumulation_steps
                        loss.backward() 
                    
                    if phase == "train" and ((index + 1) % gradient_accumulation_steps == 0 or (index + 1) == len(data_loader[phase])):
                        optimizer.step()
                        optimizer.zero_grad()
                        
                    if phase == "test": 
                        metric.update(preds, labels) 
                        
                        
                total_count += labels.shape[0]
                running_corrects += torch.sum(preds == labels.data).item() 
                
            epoch_accuracy = running_corrects / total_count

            logger.info(f"\tphase: {phase}, epoch acc: {round(epoch_accuracy, 2)}, running corrects: {running_corrects}, total checked: {total_count}, current learning rate: {round(scheduler.get_last_lr()[0], 8)}")
            
            if phase == "train": 
                train_accuracy = epoch_accuracy
            
            if phase == "test": 
                test_accuracy = epoch_accuracy
                logger.info(f"Confusion matrix:{metric.compute()}")

            if phase == 'test' and epoch_accuracy > best_accuracy:
                logger.info(f"New best model! Accuracy is {epoch_accuracy}. Saving the model.")
                best_accuracy = epoch_accuracy
                save_model(model, epoch, parameters, parameters["run path"]) 

        writer.add_scalars("Accuracy", {
            "train": train_accuracy, 
            "test": test_accuracy, 
            "best test acc": best_accuracy}, 
                           epoch)
        writer.add_scalar("Learning rate", scheduler.get_last_lr()[0], epoch)
        scheduler.step() 
        
    logger.info("Completed model training.")

    
def prepare_model(data_loaders: dict, parameters: dict) -> None:  
    logger = get_logger(__name__)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    num_classes = 2 
    
    if parameters["model"] == "efficientnet": 
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1, classes=num_classes) 
        model.classifier[1] = nn.Linear(in_features=1280, out_features=2) 
    
    if parameters["model"] == "convnext": 
        model = models.convnext_small(weights=torchvision.models.ConvNeXt_Small_Weights.IMAGENET1K_V1) 
        model.classifier[2] = nn.Sequential(nn.Linear(768, 2), 
                                            nn.Sigmoid())
    
    if parameters["model"] == "inception": 
        model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1) 
        model.aux_logits=False
        model.fc = nn.Linear(2048, num_classes)
    
    if parameters["model"] == "resnet152": 
        model = models.resnet152(weights=torchvision.models.ResNet152_Weights.DEFAULT)
        fc_layer_output = model.fc.in_features
        model.fc = torch.nn.Linear(fc_layer_output, num_classes)
    
    if parameters["model"] == "resnet50": 
        model = models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        fc_layer_output = model.fc.in_features
        model.fc = torch.nn.Linear(fc_layer_output, num_classes)
    
    model.to(device) 

    loss_function = nn.CrossEntropyLoss()
    
    optimizer = optim.SGD(model.parameters(), lr=parameters["learning rate"], momentum=0.9)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=parameters["gamma"])
    
    logger.info(f"Prepared model on {device}.")
    
    train_model(model, data_loaders, device, loss_function, optimizer, scheduler, parameters) 
    