import torch
import numpy as np
from PIL import Image
import sys
sys.path.insert(0, '')
from  model_factory import ModelFactory
from tools import image_preprocessor,label_to_class,scale_bbox
from dataset import coco_labels

# Define the factory function for instance segmentation using Mask R-CNN and YOLACT models with postprocessing
def semantic_segmentation(image_path,model_name):
    model = ModelFactory.create_semantic_model(model_name)

    input_batch,_,image_size = image_preprocessor(image_path)
    print("image shape",image_size)
    if torch.cuda.is_available():
        input_batch = input_batch.cuda()
        print("using gpu")
    
    with torch.no_grad():
        output = model(input_batch)['out'][0]
    print("output shape",output.shape)
    nmcl = output[0]
    result = postprocess(output)
    return result,nmcl

# Define function for postprocessing
def postprocess(output):
    output = output.argmax(0)
    print("post processor shape",output.shape)
    return output