import torch
from dataset import imagenet_labels
import torchvision.models as models
from torchvision.transforms import transforms
from PIL import Image

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
def classification(image_path, model_name):
    # Load the image

    # Get the model
    model = get_model(model_name)
    # Use the model for inference
# Load the image
    image = Image.open(image_path)
    # Preprocess the image
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    tensor = preprocess(image).unsqueeze(0)

    # Use the model for inference
    output = model(tensor)

    # Post-process the output
    label = post_processor(output)
    return label



# Define the post-processor function to convert class result to human read format
def post_processor(output):
    # Get the index of the predicted class
    _, index = torch.max(output, 1)
    # Convert the index to a human-readable label
    label = imagenet_labels[index[0]]
    return label


