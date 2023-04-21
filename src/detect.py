# Import necessary libraries
import torch
from torchvision.models import detection


# Define function to detect with RetinaNet
def detect_with_retinanet(image):
    # Load RetinaNet model in inference mode
    model = detection.retinanet_resnet50_fpn(pretrained=True, pretrained_backbone=True)
    model.eval()
    # Preprocess image
    image = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0)
    # Perform detection
    with torch.no_grad():
        detections = model(image)
    detections = model.postprocess(detections)
    return detections

# Define function to detect with FasterRCNN
def detect_with_fasterrcnn(image):
    # Load FasterRCNN model in inference mode
    model = detection.fasterrcnn_resnet50_fpn(pretrained=True, pretrained_backbone=True)
    model.eval()
    # Preprocess image
    image = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0)
    # Perform detection
    with torch.no_grad():
        detections = model(image)
    detections = model.postprocess(detections)
    return detections
    
# Define function to detect with SSD Lite
def detect_with_ssd_lite(image):
    # Load SSD Lite model in inference mode
    model = detection.ssd_lite_mobilenet_v3_large(pretrained=True, pretrained_backbone=True)
    model.eval()
    # Preprocess image
    image = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0)
    # Perform detection
    with torch.no_grad():
        detections = model(image)
    detections = model.postprocess(detections)
    return detections
    
# Define function to detect with Yolov3
def detect_with_yolov3(image):
    # Load Yolov3 model in inference mode
    model = torch.hub.load('ultralytics/yolov3', 'yolov3', pretrained=True)
    model.eval()
    # Preprocess image
    image = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0)
    # Perform detection
    with torch.no_grad():
        detections = model(image)
    detections = model.postprocess(detections)
    return detections


