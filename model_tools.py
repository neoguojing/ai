import typer
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
 
# logger to capture errors, warnings, and other information during the build and inference phases
TRT_LOGGER = trt.Logger()

pwd = os.getcwd()

#model1 = torch.load("hybridnets.pth",map_location=torch.device('cpu'))

# model = torch.hub.load('datvuthanh/hybridnets', 'hybridnets', pretrained=True)

# features, regression, classification, anchors, segmentation = model(img)


app = typer.Typer()

input1 = torch.randn(1,3,640,384)

@app.command()
def run_pytorch_model(path: str):
    model = load_model(path)
    if model == None:
        return
    input1 = features.get_feature_input()
    output_list = model(input1)
    print(output_list)
    

    
    
@app.command()
def run_onnx_model(path: str):
    load_model(path=path,where='onnx')
    ort_session = ort.InferenceSession(path,providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']) # 创建一个推理session
    input1 = features.get_feature_input()
    outputs = ort_session.run(None,{ 'input_0' : input1.cpu().numpy()})
    print(outputs)


def valid_check(output1: object,output2: object):
    np.testing.assert_allclose(output1.cpu().numpy(), output2[0], rtol=1e-03, atol=1e-05)
    
@app.command()
def pt_to_onnx(path: str):
    model= features.get_feature_modes()
    input_names = ["input_0"]
    output_names = ["output_0"]
    input1 = torch.randn(1, 3, 224, 224)
    if torch.cuda.is_available():
        input1 = input1.cuda()
    torch.onnx.export(model,input1,path+'.onnx',export_params=True,do_constant_folding=True,
                         verbose = True,input_names=input_names,output_names=output_names,opset_version=11)
    
@app.command()
def onnx_to_tensorrt(path: str):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network()
    parser = trt.OnnxParser(network, TRT_LOGGER)
     
    # parse ONNX
    with open(path, 'rb') as model:
        print('Beginning ONNX file parsing')
        parser.parse(model.read())
    print('Completed parsing of ONNX file')
    # allow TensorRT to use up to 1GB of GPU memory for tactic selection
    # builder.max_workspace_size = 1 << 30
    # we have only one image in batch
    builder.max_batch_size = 1
    # use FP16 mode if possible
    if builder.platform_has_fast_fp16:
        builder.fp16_mode = True
    print('Building an engine...')
    engine = builder.build_cuda_engine(network)
    context = engine.create_execution_context()
    print("Completed creating Engine")
 
    return engine, context

def run_on_tensorrt(engine,context):
    # get sizes of input and output and allocate memory required for input data and for output data
    for binding in engine:
        if engine.binding_is_input(binding):  # we expect only one input
            input_shape = engine.get_binding_shape(binding)
            input_size = trt.volume(input_shape) * engine.max_batch_size * np.dtype(np.float32).itemsize  # in bytes
            device_input = cuda.mem_alloc(input_size)
        else:  # and one output
            output_shape = engine.get_binding_shape(binding)
            # create page-locked memory buffers (i.e. won't be swapped to disk)
            host_output = cuda.pagelocked_empty(trt.volume(output_shape) * engine.max_batch_size, dtype=np.float32)
            device_output = cuda.mem_alloc(host_output.nbytes)
    
    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()
    host_input = np.array(preprocess_image("turkish_coffee.jpg").numpy(), dtype=np.float32, order='C')
    cuda.memcpy_htod_async(device_input, host_input, stream)
    
    context.execute_async(bindings=[int(device_input), int(device_output)], stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(host_output, device_output, stream)
    stream.synchronize()
    
    output_data = torch.Tensor(host_output).reshape(engine.max_batch_size, output_shape[0])
    postprocess(output_data)

    
def load_model(path: str,where: str = 'jit'):
    model = object()
    if where == 'jit':
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            model = torch.jit.load(path)
            # 关闭用于训练和评估的Dropouts（防止过拟合），BatchNorm（归一化，加速训练）
            model.eval().cuda()
        else:
            model = torch.jit.load(path,map_location=torch.device('cpu'))
            model.eval()
    elif where == 'pytorch':
        checkpoint = torch.load(PATH)
        print(checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
    elif where == 'onnx':
        model = onnx.load(path) # 加载onnx
        onnx.checker.check_model(model)
        print(onnx.helper.printable_graph(model.graph))
        for _input in model.graph.input:
            print(MessageToDict(_input))
            
    return model

@app.command()
def save_model(name: str = 'resnet18',dst: str = pwd ):
    model = features.get_feature_modes()
    model_scripted = torch.jit.script(model) # Export to TorchScript
    name = os.path.join(dst,name+'.pt')
    print(name)
    model_scripted.save(name) # Save
     
@app.command()
def info(path: str,where: str = 'jit'):
    if where != 'jit':
        model = load_model(path,where)
        if model == None:
            return

        print('input shape:',model.input_shape)
        print('input size:',model.size)
        print('params:',model.parameters())

        for  layer in model:
            x = layer(input1)
            print(layer.__class__.__name__,"output shape:",x.shape)
            
def print_model_info(model: object):
    print('params:',model.parameters())
    
def image_format(path: str,size,shape):
    img = cv2.imread(path)
    img = cv2.resize(img,size)
    img = np.reshape(img,shape)
    return img

        
@app.command()
def postprocess(output_data,class_file="imagenet_classes.txt",threshhold=0.5):
    # get class names
    with open(class_file) as f:
        classes = [line.strip() for line in f.readlines()]
    # calculate human-readable value by softmax
    confidences = torch.nn.functional.softmax(output_data, dim=1)[0] * 100
    # find top predicted classes
    _, indices = torch.sort(output_data, descending=True)
    i = 0
    # print the top classes predicted by the model
    while confidences[indices[0][i]] > threshhold:
        class_idx = indices[0][i]
        print(
            "class:",
            classes[class_idx],
            ", confidence:",
            confidences[class_idx].item(),
            "%, index:",
            class_idx.item(),
        )
        i += 1

@app.command()
def download(git: str,name: str,where: str = 'pytorch'):
    if where == "pytorch":
        model = torch.hub.load(git, name, pretrained=True)
        model_scripted = torch.jit.script(model) # Export to TorchScript
        model_scripted.save(pwd+name+'.pt') # Save
    
@app.command()
def test():
    print("test")
    
if __name__ == '__main__':
    app()