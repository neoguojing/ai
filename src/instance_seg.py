# Define the function for Mask R-CNN model
# Import the necessary libraries
import torch
import numpy as np
import sys
sys.path.insert(0, '')
from  model_factory import ModelFactory
from tools import image_preprocessor,label_to_class,scale_bbox
from dataset import coco_labels

# Define the factory function for instance segmentation using Mask R-CNN and YOLACT models with postprocessing
def instance_segmentation(image_path,model_name):
    model = ModelFactory.create_instance_model(model_name)

    input_batch,scale_factor = image_preprocessor(image_path)
    if torch.cuda.is_available():
        input_batch = input_batch.cuda()
        print("using gpu")
    
    with torch.no_grad():
        output = model(input_batch)
        
    boxes,labels,masks = postprocess(output,scale_factor)
    return boxes,labels,masks

def postprocess(output,scale_factor):
    # Perform postprocessing on the model output
    # get the predicted boxes, labels, and masks for the objects in the image
    boxes = output[0]['boxes'].detach().numpy()
    new_boxes = scale_bbox(bboxs=boxes,factor=scale_factor)
    labels = output[0]['labels'].detach().numpy()
    classs = label_to_class(labels,coco_labels)

    masks = output[0]['masks'].detach().numpy()

    print("boxes:",boxes)
    print("new_boxes:",new_boxes)
    print("labels:",labels)
    print("classs:",classs)
    print("masks:",masks)
    return new_boxes,classs,masks