# 使用官方的 Python 基础镜像
FROM python:3.12.1

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

ENV PIP_NO_CACHE_DIR=off
# 设置环境变量
ENV PYTHONUNBUFFERED=1

# 安装 Python 依赖包
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 暴露端口（如果需要）
EXPOSE 7860

# 运行项目
CMD ["python", "app.py"]
