from torchvision import models
import numpy as np
from pycuda import driver as cuda
from torchvision.models import detection
import utils
import torch
import torchvision
import tensorrt
import tensorrt as trt
import onnx
import onnxruntime as ort

class ModelFactory:
    
    MODELS_FEATURE_EXTRACT = {
        'resnet50': models.resnet50(weight=utils.ResNet18_Weights.DEFAULT),
        'vgg16': models.vgg16(pretrained=True),
        'inception_v3': models.inception_v3(pretrained=True),
        'mobilenet_v2': models.mobilenet_v2(pretrained=True),
        'densenet121': models.densenet121(pretrained=True)
    }

    MODELS_DETECT = {
        'RetinaNet': detection.retinanet_resnet50_fpn(pretrained=True, pretrained_backbone=True),
        'FasterRCNN': detection.fasterrcnn_resnet50_fpn(pretrained=True, pretrained_backbone=True),
        'SSDLite': detection.ssd300_vgg16(pretrained=True),
        'Yolov5': torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    }

    MODELS_CLASSIFICATION = {
        'resnet50': models.resnet50(pretrained=True),
        'mobilenetv2': models.mobilenet_v2(pretrained=True),
        'shufflenetv2': models.shufflenet_v2_x1_0(pretrained=True)
    }

    MODELS_INSTANCE = {
        'maskrcnn': torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True),
        'yolact': torch.hub.load('dbolya/yolact', 'yolact_resnet50', pretrained=True)
    }

    MODELS_SEMANTIC = {
        'pspnet': torchvision.models.segmentation.pspnet(pretrained=True),
        'deeplabv3': torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True),
        'bisenetv1': torch.hub.load('catalyst-team/deeplabv3', 'deeplabv3_resnet50', pretrained=True)
    }

    @staticmethod    
    def create_feature_extract_model(model_name):
        if model_name not in ModelFactory.MODELS_FEATURE_EXTRACT:
            raise ValueError('Invalid model name')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = ModelFactory.MODELS_FEATURE_EXTRACT[model_name].to(device)
        model.eval()
        return model
    
    @staticmethod  
    def create_detect_model(model_name):
        if model_name not in ModelFactory.MODELS_DETECT:
            raise ValueError('Invalid model name')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = ModelFactory.MODELS_DETECT[model_name].to(device)
        model.eval()
        return model

    @staticmethod
    def create_classication_model(model_name):
        if model_name not in ModelFactory.MODELS_CLASSIFICATION:
            raise ValueError('Invalid model name')
        # Use GPU if available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = ModelFactory.MODELS_CLASSIFICATION[model_name].to(device)
        model.eval() # Set model to inference mode
        return model

    @staticmethod
    def create_instance_model(model_name):
        if model_name not in ModelFactory.MODELS_INSTANCE:
            raise ValueError('Invalid model name')
        # Use GPU if available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = ModelFactory.MODELS_INSTANCE[model_name].to(device)
        model.eval() # Set model to inference mode
        return model

    @staticmethod
    def create_semantic_model(model_name):
        if model_name not in ModelFactory.MODELS_SEMANTIC:
            raise ValueError('Invalid model name')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = ModelFactory.MODELS_SEMANTIC[model_name].to(device)
        model.eval()
        return model

    @staticmethod
    def convert_model(model, output_path):
        """
        This method converts a model to ONNX format, saves it to the specified output path, and converts it to TensorRT format.
        :param model: An instance of a model to convert.
        :param output_path: A string representing the path to save the converted model.
        :return: None
        """
        # Convert model to ONNX format
        onnx_model = onnx.load(model)
        
        # Save ONNX model to output path
        onnx.save(onnx_model, output_path)
        
        # Convert ONNX model to TensorRT format
        trt_model = tensorrt.convert_onnx_model(onnx_model)
        
        # Save TensorRT model to output path
        with open(output_path, "wb") as f:
            f.write(trt_model)

    @staticmethod
    def load_and_run_trt_model(model_path):
        """
        This method loads a TensorRT model from the specified path and runs it.
        :param model_path: A string representing the path to the TensorRT model.
        :return: None
        """
        with open(model_path, "rb") as f:
            trt_model = f.read()
        runtime = tensorrt.Runtime(tensorrt.Logger())
        engine = runtime.deserialize_cuda_engine(trt_model)
        context = engine.create_execution_context()
        # Run inference on the model
        input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
        output_data = np.empty((1, 1000), dtype=np.float32)
        # Allocate device memory for inputs and outputs
        d_input = torch.cuda.mem_alloc(1 * input_data.nbytes)
        d_output = torch.cuda.mem_alloc(1 * output_data.nbytes)
        # Create a stream to run inference
        stream = torch.cuda.Stream()
        # Transfer input data to device
        torch.cuda.memcpy_htod_async(d_input, input_data, stream)
        # Run inference
        context.enqueue(1, [int(d_input), int(d_output)])
        # Transfer predictions back from device
        torch.cuda.memcpy_dtoh_async(output_data, d_output, stream)
        # Synchronize the stream
        stream.synchronize()
        return output_data

    @staticmethod
    def load_and_run_onnx_model(model_path, input_data):
        """
        This method loads an ONNX model from the specified path, runs it with the given input data, and returns the output.
        :param model_path: A string representing the path to the ONNX model.
        :param input_data: A numpy array representing the input data for the model.
        :return: A numpy array representing the output of the model.
        """
        # Load ONNX model from path
        onnx_model = onnx.load(model_path)
        
        # Create TensorRT engine from ONNX model
        trt_logger = tensorrt.Logger()
        trt_builder = tensorrt.Builder(trt_logger)
        trt_network = trt_builder.create_network()
        trt_parser = trt.OnnxParser(trt_network, trt_logger)
        trt_parser.parse(onnx_model.SerializeToString())
        trt_builder.max_batch_size = 1
        trt_builder.max_workspace_size = 1 << 30
        trt_engine = trt_builder.build_cuda_engine(trt_network)
        
        # Allocate device memory for inputs and outputs
        input_shape = trt_engine.get_binding_shape(0)
        output_shape = trt_engine.get_binding_shape(1)
        d_input = torch.cuda.mem_alloc(input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3] * 4)
        d_output = torch.cuda.mem_alloc(output_shape[0] * output_shape[1] * 4)
        
        # Create a stream to run inference
        stream = torch.cuda.Stream()
        
        # Transfer input data to device
        torch.cuda.memcpy_htod_async(d_input, input_data.ravel().astype(np.float32), stream)
        
        # Run inference
        context = trt_engine.create_execution_context()
        context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
        
        # Transfer predictions back from device
        output_data = np.empty(output_shape, dtype=np.float32)
        torch.cuda.memcpy_dtoh_async(output_data, d_output, stream)
        
        # Synchronize the stream
        stream.synchronize()
        
        return output_data








        
          

