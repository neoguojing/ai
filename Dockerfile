# Use an official Python runtime as a parent image
# FROM guojingneo/pytorch-tensorflow-notebook:latest
FROM guojingneo/pytorch-notebook:latest

# RUN pip install cython
# RUN pip install pycocotools
RUN pip install -qr https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt
# RUN git clone https://github.com/dbolya/yolact.git 
# RUN cd yolact && conda env create -f environment.yml \
#     && wget https://drive.google.com/file/d/1yp7ZbbDwvMiFJEq4ptVKTYTI2VeRDXl0/view?usp=sharing -fR -O yolact_resnet50.pth  

# Set the working directory to /app
WORKDIR /workspace

ENV DATASET_PREFIX=/workspace/dataset/

# Copy the current directory contents into the container at /app
# COPY . /workspace

# Install any needed packages specified in requirements.txt
# RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80


# Run app.py when the container launches
# CMD ["python", "app.py"]
CMD ["chown", "-R user:user /workspace"]
