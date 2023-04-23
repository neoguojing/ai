# Import necessary libraries
import torch
import torchvision
import numpy as np
from PIL import Image

# Define model factory function
def get_model(model_name):
    if model_name == 'pspnet':
        model = torchvision.models.segmentation.pspnet(pretrained=True)
    elif model_name == 'deeplabv3':
        model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
    elif model_name == 'bisenetv1':
        model = torch.hub.load('catalyst-team/deeplabv3', 'deeplabv3_resnet50', pretrained=True)
    else:
        raise ValueError('Invalid model name')
    model.eval()
    return model


# Define function for postprocessing
def postprocess(output):
    output = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy()
    return output