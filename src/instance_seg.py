# Define the function for Mask R-CNN model
# Import the necessary libraries
import torch
import numpy as np
import sys
sys.path.insert(0, '')
from  model_factory import ModelFactory
from tools import image_preprocessor,label_to_class,scale_bbox
from dataset import coco_labels
import cv2

# Define the factory function for instance segmentation using Mask R-CNN and YOLACT models with postprocessing
def instance_segmentation(image_path,model_name):
    model = ModelFactory.create_instance_model(model_name)

    input_batch,_,image_size = image_preprocessor(image_path)
    print("image shape",image_size)
    if torch.cuda.is_available():
        input_batch = input_batch.cuda()
        print("using gpu")
    
    with torch.no_grad():
        output = model(input_batch)[0]
#     print(output)
    result = postprocess(output,image_size)
    return result

def postprocess(output,img_shape,threshold=0.1,max_detections=100):
    scores = output['scores'].detach()
    mask = scores > threshold
    scores = scores[mask].detach()
    boxes = output['boxes'][mask].detach()
    labels = output['labels'][mask].detach()
    classs = label_to_class(labels,coco_labels)
    masks = output['masks'][mask].detach()
    
    # Sort the predictions by confidence scores and keep the top-k detections
    _, indices = scores.sort(descending=True)
    indices = indices[:max_detections]
    scores = scores[indices]
    boxes = boxes[indices]
    labels = labels[indices]
    masks = masks[indices]
    print(masks.shape)
    # Resize masks to the original image size
    mask_h, mask_w = masks.shape[-2:]
    masks = masks.permute(0, 2, 3, 1).contiguous().view(-1, mask_h * mask_w)
    masks = torch.sigmoid(masks) > 0.5
    masks = masks.view(-1, mask_h, mask_w, 1).permute(0, 3, 1, 2)
    print(masks.shape)
    return {
        'boxes': boxes,
        'labels': labels,
        'scores': scores,
        'masks': masks,
        'classes': classs,
    }