# ai

## install tensor RT
- pip install nvidia-pyindex
- pip install nvidia-tensorrt
- pip3 install pycuda 

## 生成文件

pip freeze >requirements.txt



```
docker build -t guojingneo/pytorch-tensorflow-notebook .
```

```
docker run --gpus all -p 8888:8888 --rm -it --name pt_tf_notebook pytorch-tensorflow-notebook
```

