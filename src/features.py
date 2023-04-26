import torch
import sys
sys.path.insert(0, '')
from model_factory import ModelFactory
from tools import image_preprocessor


def extract_features(image_path, model_name):
    model = ModelFactory.create_feature_extract_model(model_name)

    input_batch,_ = image_preprocessor(image_path)

    if torch.cuda.is_available():
        input_batch = input_batch.cuda()
        print("using gpu")

    with torch.no_grad():
        output = model(input_batch)
        
    return output



    
