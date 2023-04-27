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
    
    print(output)
    boxes,labels,masks = postprocess(output,image_size)
    return boxes,labels,masks

def postprocess(output,img_shape,threshold=0.5):
    # Perform postprocessing on the model output
    # get the predicted boxes, labels, and masks for the objects in the image
    boxes = output[0]['boxes'].detach().cpu().numpy()
    labels = output[0]['labels'].detach().cpu().numpy()
    classs = label_to_class(labels,coco_labels)
    masks = output[0]['masks'].detach().cpu().numpy()
    scores = output[0]['scores'].detach().cpu().numpy()
    
    print(img_shape)
    print(masks.shape[0])
    # Resize the masks to the size of the input image
    resized_masks = np.zeros((masks.shape[0], img_shape[1], img_shape[0], 1))
    for i in range(masks.shape[0]):
#         mask = masks[i, :, :, 0]
        mask = masks[i]
        print("masks[i]",masks[i])
        resized_mask = cv2.resize(mask,(1008, 1800), interpolation=cv2.INTER_LINEAR)
        print("resized_mask",resized_mask)
        resized_mask = np.expand_dims(resized_mask, axis=-1)
        resized_masks[i, :, :, :] = resized_mask


    # Filter out masks with low confidence
    
    keep = np.where(scores > threshold)[0]
    masks = masks[keep]
    resized_masks = resized_masks[keep]
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
    print("resized_masks:",resized_masks)
    return boxes,classs,masks