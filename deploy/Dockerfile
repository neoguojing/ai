
# Start with an official NVIDIA CUDA image
FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    libgl1-mesa-dev \
    libglib2.0-0 \ 
 && rm -rf /var/lib/apt/lists/*

 RUN dpkg-reconfigure --frontend noninteractive tzdata

# Create a working directory
RUN mkdir /workspace
WORKDIR /workspace

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /workspace
USER user

# Install Miniconda and Python 3.8
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && bash Miniconda3-latest-Linux-x86_64.sh -b -p /home/user/miniconda \
 && rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/home/user/miniconda/bin:${PATH}
RUN conda update -y conda \
 && conda install -y python=3.8

# Install PyTorch, TensorFlow, and Jupyter Notebook
RUN conda install -y -c pytorch pytorch torchvision torchaudio -c=pytorch-lts \
#  && conda install -y -c anaconda tensorflow-gpu \
 && conda install -y -c anaconda jupyter

# Set up the entry point for Jupyter Notebook
EXPOSE 8888
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
