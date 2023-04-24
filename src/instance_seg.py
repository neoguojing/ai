# Define the function for Mask R-CNN model
# Import the necessary libraries
import torch
import torchvision
from PIL import Image
from torchvision import transforms

def get_model(model_name):
    if model_name == 'maskrcnn':
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    elif model_name == 'yolact':
        model = torch.hub.load('dbolya/yolact', 'yolact_resnet50', pretrained=True)
    else:
        raise ValueError('Invalid model name')
    # Use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval() # Set model to inference mode
    return model


# Define the function for postprocessing
def postprocess(output):
    # Perform postprocessing on the model output
    # get the predicted boxes, labels, and masks for the objects in the image
    boxes = output[0]['boxes'].detach().numpy()
    labels = output[0]['labels'].detach().numpy()
    masks = output[0]['masks'].detach().numpy()

    result = {boxes,labels,masks}

    return result

# Define the function for instance segmentation using Mask R-CNN and YOLACT models with postprocessing
# Define the factory function for instance segmentation using Mask R-CNN and YOLACT models with postprocessing
def instance_segmentation(image_path, model_name):
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    image = transform(image)
    if torch.cuda.is_available():
        image = image.to('cuda')
    
    model = get_model(model_name)
    output = None
    with torch.no_grad():
        output = model([image])
        
    postprocessed_output = postprocess(output)
    return postprocessed_output

