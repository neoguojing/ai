import typer
import torch
import onnx
import onnxruntime as ort
from importlib import import_module
import cv2
import numpy as np
import os
import features
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
    output_list = model(img)
    print(output_list)
    
    
    
@app.command()
def run_onnx_model(path: str):
    model = load_model(path,'onnx')
    ort_session = ort.InferenceSession(path) # 创建一个推理session
    outputs = ort_session.run(None,{ 'input_0' : input1})
    print(outputs)


def valid_check(output1: object,output2: object):
    np.testing.assert_allclose(output1.cpu().numpy(), output2[0], rtol=1e-03, atol=1e-05)
    
@app.command()
def pt_to_onnx(path: str):
    model = load_model(path)
    if model == None:
        return
    
    input_names = ["input_0"]
    output_names = ["output_0"]
    
    onnx_model = torch.onnx.export(model,input1,path+'onnx',input_names=input_names,output_names=output_names)
    onnx.checker.check_model(onnx_model)
    
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
        if not onnx.checker.check_model(model): # 检查生成模型是否错误
            return None
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
    # print('input shape:',model.inpu)
    # print('input size:',model.size)
    print('params:',model.parameters())
    
def image_format(path: str,size,shape):
    img = cv2.imread(path)
    img = cv2.resize(img,size)
    img = np.reshape(img,shape)
    return img
    
@app.command()
def download(git: str,name: str,where: str = 'pytorch'):
    if where == "pytorch":
        model = torch.hub.load(git, name, pretrained=True)
        model_scripted = torch.jit.script(model) # Export to TorchScript
        model_scripted.save(pwd+name+'.pt') # Save
    
@app.command()
def test():
    model = features.get_feature_modes()
    print_model_info(model)
if __name__ == '__main__':
    app()