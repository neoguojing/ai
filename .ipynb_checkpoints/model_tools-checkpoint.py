import typer
import torch
import onnx
import onnxruntime as ort
from importlib import import_module
import cv2
import numpy as np

#model1 = torch.load("hybridnets.pth",map_location=torch.device('cpu'))

# model = torch.hub.load('datvuthanh/hybridnets', 'hybridnets', pretrained=True)

# features, regression, classification, anchors, segmentation = model(img)


app = typer.Typer()

model_dir = '/data/ai/models/'
input1 = torch.randn(1,3,640,384)

@app.command()
def run_pytorch_model(path: str, gpu: bool = False):
    model = load_model(path,gpu)
    if model == None:
        return
    output_list = model(img)
    print(output_list)
    
    
    
@app.command()
def run_onnx_model(path: str, gpu: bool = False):
    model = load_model(path,gpu,'onnx')
    ort_session = ort.InferenceSession(path) # 创建一个推理session
    outputs = ort_session.run(None,{ 'input_0' : input1})
    print(outputs)


def valid_check(output1: object,output2: object):
    np.testing.assert_allclose(output1.cpu().numpy(), output2[0], rtol=1e-03, atol=1e-05)
    
@app.command()
def pth_to_onnx(path: str,gpu: bool = False):
    model = load_model(path,gpu)
    if model == None:
        return
    
    input_names = ["input_0"]
    output_names = ["output_0"]
    
    onnx_model = torch.onnx.export(model,input1,path+'onnx',input_names=input_names,output_names=output_names)
    onnx.checker.check_model(onnx_model)
    
def load_model(path: str,gpu: bool,where: str = 'pytorch'):
    model = object()
    if where == 'jit':
        if gpu:
            use_gpu = torch.cuda.is_available()
            if not use_gpu:
                print("unsupport gpu")
                return None
            input1 = input1.cuda()
            model = torch.jit.load(path)
            # 关闭用于训练和评估的Dropouts（防止过拟合），BatchNorm（归一化，加速训练）
            model.eval().cuda()
        else:
            model = torch.jit.load(path,map_location=torch.device('cpu'))
            model.eval()
    else if where == 'pytorch':
        checkpoint = torch.load(PATH)
        print(checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
    else if where == 'onnx':
        model = onnx.load(path) # 加载onnx
        onnx.checker.check_model(model) # 检查生成模型是否错误
    return model
    
@app.command()
def info(path: str,gpu: bool = False,where: str = 'pytorch'):
    if where == 'pytorch':
        model = load_model(path,gpu,where)
        if model == None:
            return

        print('input shape:',model.input_shape)
        print('input size:',model.size)
        print('params:',model.parameters())

        for  layer in model:
            x = layer(input1)
            print(layer.__class__.__name__,"output shape:",x.shape)

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
        model_scripted.save(model_dir+name+'.pt') # Save
    
if __name__ == '__main__':
    app()