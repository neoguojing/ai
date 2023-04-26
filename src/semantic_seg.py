# Import necessary libraries
import torch

# Define function for postprocessing
def postprocess(output):
    output = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy()
    return output