# Define variables for the image name and tag
IMAGE_NAME = guojingneo/ai-world
IMAGE_TAG = latest

BASE_IMAGE_NAME = guojingneo/pytorch-tensorflow-notebook
BASE_IMAGE_TAG = latest

# Define the build command
build:
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .

# Define the clean command
clean:
	docker rmi $(IMAGE_NAME):$(IMAGE_TAG)

# Define the push command
push:
	docker push $(IMAGE_NAME):$(IMAGE_TAG)

# Define the run command
run:
	docker run --gpus all -p 8888:8888 -v $(pwd):/workspace:rw --rm -it --name ai-world $(IMAGE_NAME):$(IMAGE_TAG)

# Define the base command
base:
	docker build -t $(BASE_IMAGE_NAME):$(BASE_IMAGE_TAG) ./deploy/
	docker push $(BASE_IMAGE_NAME):$(BASE_IMAGE_TAG)

