import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


def get_feature_modes():
    model = models.resnet18(pretrained=True)
    input_image = Image.open('/data/ai/ai/pexels-pixabay-45201.jpg')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.cuda()
        model.eval().cuda()

    # with torch.no_grad():
    #     output = model(input_batch)
    # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
    # print('out:',output[0])
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    # probabilities = torch.nn.functional.softmax(output[0], dim=0)
    # print('probabilities:',probabilities)
    return model

def get_feature_input():
    model = models.resnet18()
    input_image = Image.open('/data/ai/ai/pexels-pixabay-45201.jpg')
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
    print(input_tensor)
    print(input_batch)
    return input_batch
    
