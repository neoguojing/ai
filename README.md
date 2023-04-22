# ai

## install tensor RT
- pip install nvidia-pyindex
- pip install nvidia-tensorrt
- pip3 install pycuda 

## 生成文件

pip freeze >requirements.txt


## compile base docker image
```
docker build -t guojingneo/pytorch-tensorflow-notebook ./deploy
```
## run base docker image
```
docker run --gpus all -p 8888:8888 --rm -it --name pt_tf_notebook guojingneo/pytorch-tensorflow-notebook

```

## compile app docker image
```
docker build -t guojingneo/ai-world .
```
## run app docker image
```

docker run --gpus all -p 8888:8888 -v $(pwd):/workspace:rw --rm -it --name ai-world guojingneo/ai-world 

```

