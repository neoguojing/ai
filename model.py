import torch
import onnx
import onnxruntime as ort
from importlib import import_module
import cv2
import numpy as np
import os
import features
import torchvision.models as models
from PIL import Image
from google.protobuf.json_format import MessageToDict
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
from torch2trt import torch2trt



class Model(object):
    model = None
    
    def __init__(self，path，model_type):
        self.model = load_model_from_file()
        self.path = path
        self.model_type = model_type
        
        
    @classmethod
    def init_by_obj(self, model):
        self.model = model
        
    
    def load_model_from_file(path，model_type):
        

class PthModel(Model):
    def __init__(self，path，model_type):
        
    @classmethod
    def init_by_obj(self, model):
        self.model = model
        
    
    def load_model_from_file(path，model_type):
        
    
    def to_trt(model,inputs):
        model_trt = torch2trt(model, [inputs])
        return TrtModel.init_by_obj(model_trt)

class OnnxModel(Model):
    def __init__(self，path，model_type):
        
    def to_trt(model,inputs):
        model_trt = torch2trt(model, [inputs])
        return TrtModel.init_by_obj(model_trt)
    
class TrtModel(Model):
    def __init__(self，path，model_type):
        
    @classmethod
    def init_by_obj(self, model):
        self.model = model