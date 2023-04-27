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
    if torch.cuda.is_available():
        input_batch = input_batch.cuda()
        print("using gpu")
    
    with torch.no_grad():
        output = model(input_batch)
        
    boxes,labels,masks = postprocess(output,image_size)
    return boxes,labels,masks

def postprocess(output,img_shape,threshold=0.5):
    # Perform postprocessing on the model output
    # get the predicted boxes, labels, and masks for the objects in the image
    boxes = output[0]['boxes'].detach().numpy()
    labels = output[0]['labels'].detach().numpy()
    classs = label_to_class(labels,coco_labels)
    masks = output[0]['masks'].detach().numpy()

    # Resize the masks to the size of the input image
    masks = np.transpose(masks, (1, 2, 0))
    masks = cv2.resize(masks, img_shape[:2][::-1], interpolation=cv2.INTER_LINEAR)
    masks = np.transpose(masks, (2, 0, 1))

    # Filter out masks with low confidence
    scores = output['scores'].detach().cpu().numpy()
    keep = np.where(scores > threshold)[0]
    masks = masks[keep]
    boxes = boxes[keep]
    labels = labels[keep]
    classs = classs[keep]
    scores = scores[keep]

    # # Apply non-maximum suppression to the boxes
    # keep = []
    # while boxes.shape[0] > 0:
    #     i = np.argmax(scores)
    #     keep.append(i)
    #     overlap = bbox_iou(boxes[i], boxes)
    #     inds = np.where(overlap <= 0.5)[0]
    #     boxes = boxes[inds]
    #     masks = masks[inds]
    #     labels = labels[inds]
    #     scores = scores[inds]
    # boxes = output_boxes[keep]
    # masks = output_masks[keep]
    # labels = output_labels[keep]


    print("boxes:",boxes)
    print("labels:",labels)
    print("classs:",classs)
    print("masks:",masks)
    return boxes,classs,masks