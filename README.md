# Classification reference model

## Overview 

Similar to my YOLO code, I have created a set of PyTorch code that can be easily adapted to different classification tasks. Configuration can be done using the YAML file under ```parameters```. All hyperparameters and configurations are saved under ```runs```, which are incremented one-by-one. The code uses TensorBoard for tracking, and has gradient accumulation built-in. 

## Configuration 

An example configuration file is shown below. 

```yaml
run name: Testing
root: "../Datasets/Ants vs bees"

model: "resnet50"

learning rate: 0.0005
gamma: 0.9 
epochs: 1000 

desired batch size: 16
real batch size: 8 

only save best: Yes 
```

## Supported models 

I have added some network architectures for fine-tuning, with pre-trained weights downloaded from the PyTorch website. The specific weight sets (small, medium, large etc.) can be changed under ```models.py```. 

- ResNet50 
- ResNet152 
- EfficientNetV2 
- ConvNeXt 
- InceptionV3
