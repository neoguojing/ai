# 使用指定的基础镜像
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# 更新 apt-get 并安装依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends && \
    apt-get install -y git

COPY . /workspace
# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements-docker.txt

# 设置环境变量
ENV PYTHONUNBUFFERED=1

# 暴露应用的端口，如果有需要，可以在这里指定
# EXPOSE 8000

# 复制当前目录所有文件到容器的工作目录

WORKDIR /workspace/detectron/demo
# 设置容器启动时执行的命令
CMD ["python", "main.py"]
