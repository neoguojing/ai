import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models.utils as utils

def get_model(model_name):
    if model_name == 'resnet50':
        model = models.resnet50(weight=utils.ResNet18_Weights.DEFAULT)
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif model_name == 'inception_v3':
        model = models.inception_v3(pretrained=True)
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=True)
    elif model_name == 'densenet121':
        model = models.densenet121(pretrained=True)
    else:
        raise ValueError('Invalid model name')
    model.eval()
    return model

def extract_features(model, input_image_path):
    input_image = Image.open(input_image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    if torch.cuda.is_available():
        input_batch = input_batch.cuda()
        model.eval().cuda()
        print("using gpu")

    with torch.no_grad():
        output = model(input_batch)
    return output[0]





    
