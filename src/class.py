
import torch
from dataset import imagenet_labels
import torchvision.models as models
from torchvision.models import shufflenet_v2_x1_0

# Define the model factory function
def get_model(model_name):
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
    elif model_name == 'mobilenetv2':
        model = models.mobilenet_v2(pretrained=True)
    elif model_name == 'shufflenetv2':
        model = models.shufflenet_v2_x1_0(pretrained=True)
    else:
        raise ValueError('Invalid model name')
    # Use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval() # Set model to inference mode
    return model


# Define the classification function for multi-class classification
def classification(image,model):

    output = model(image)
    # Post-process the output
    post_processor(output)
    return output

# Define the post-processor function to convert class result to human read format
def post_processor(output):
    # Get the index of the predicted class
    _, index = torch.max(output, 1)
    # Convert the index to a human-readable label
    label = imagenet_labels[index[0]]
    return label







