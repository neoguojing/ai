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


class App(object):
    def __init__(self, path, model_type):
        self.model = load_model_from_file(path, model_type)
        self.path = path
        self.model_type = model_type
        self.model_trt = None
        self.model_onnx = None
        self.model_pth = None
        self.threshold = 0.5

    @classmethod
    def init_by_obj(cls, model):
        obj = cls.__new__(cls)
        obj.model = model
        obj.path = ""
        obj.model_type = None
        obj.model_trt = None
        obj.model_onnx = None
        obj.model_pth = None
        obj.threshold = 0.5
        return obj

    def load_model_from_file(self, path, model_type):
        if model_type == "pth":
            model = torch.load(path)
        elif model_type == "onnx":
            onnx_model = onnx.load(path)
            model = onnx_to_pytorch(onnx_model)
        else:
            raise ValueError("Invalid model type")
        return model

    def pth2trt(self, x):
        self.model_trt = torch2trt(self.model, [x])

        