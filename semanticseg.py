# Import necessary libraries
import torch
import torchvision
import numpy as np
from PIL import Image

# Define function for PSPNet model
def get_pspnet_model():
    model = torchvision.models.segmentation.pspnet(pretrained=True)
    model.eval()
    return model

# Define function for DeepLabV3 model
def get_deeplabv3_model():
    model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
    model.eval()
    return model

# Define function for BiSeNet v1 model with postprocessor
def get_bisenetv1_model():
    model = torch.hub.load('catalyst-team/deeplabv3', 'deeplabv3_resnet50', pretrained=True)
    model.eval()
    return model

# Define function for postprocessing
def postprocess(output):
    output = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy()
    return output