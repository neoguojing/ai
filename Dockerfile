# Use an official Python runtime as a parent image
FROM guojingneo/pytorch-tensorflow-notebook:latest

RUN apt-get install libglvnd-dev mesa-utils
RUN pip install -qr https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt

# Set the working directory to /app
WORKDIR /workspace

# Copy the current directory contents into the container at /app
# COPY . /workspace

# Install any needed packages specified in requirements.txt
# RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80


# Run app.py when the container launches
# CMD ["python", "app.py"]
