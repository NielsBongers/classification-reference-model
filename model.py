from pathlib import Path 

import torch 
import torch.nn as nn 
import torch.optim as optim 
import torchvision 
import torchvision.models as models 
from torch.utils.tensorboard import SummaryWriter

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
            
            # metric = MulticlassConfusionMatrix(num_classes=2).to(device)

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
                    
                    if phase == "train" and index % gradient_accumulation_steps == 0: 
                        optimizer.step()
                        optimizer.zero_grad()
                        
                    # if phase == "test": 
                        # metric.update(preds, labels) 
                        
                total_count += labels.shape[0]
                running_corrects += torch.sum(preds == labels.data).item() 
                
            epoch_accuracy = running_corrects / total_count

            logger.info(f"\tphase: {phase}, epoch acc: {round(epoch_accuracy, 2)}, running corrects: {running_corrects}, total checked: {total_count}, current learning rate: {round(scheduler.get_last_lr()[0], 8)}")
            
            if phase == "train": 
                train_accuracy = epoch_accuracy
            
            if phase == "test": 
                test_accuracy = epoch_accuracy
                # logger.info(f"Confusion matrix:{metric.compute()}")

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
    
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1, classes=2) 
    model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes) 
    model = model.to(device)

    loss_function = nn.CrossEntropyLoss()
    
    optimizer = optim.SGD(model.parameters(), lr=parameters["learning rate"], momentum=0.9)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=parameters["gamma"])
    
    logger.info(f"Prepared model on {device}.")
    
    train_model(model, data_loaders, device, loss_function, optimizer, scheduler, parameters) 
    