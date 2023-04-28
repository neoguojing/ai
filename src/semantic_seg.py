import torch
import numpy as np
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
        output = model(input_batch)[0]
#     print(output)
    result = postprocess(output,image_size)
    return result

# Define function for postprocessing
def postprocess(output,image_shape):
    print(output)
    return output