# 使用指定的基础镜像
FROM guojingneo/ai-base:latest

COPY . /workspace
# 设置环境变量
ENV PYTHONUNBUFFERED=1

# 暴露应用的端口，如果有需要，可以在这里指定
EXPOSE 7860

# 复制当前目录所有文件到容器的工作目录

WORKDIR /workspace/detectron/demo
# 设置容器启动时执行的命令
CMD ["python", "main.py"]
