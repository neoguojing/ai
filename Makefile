# Define variables for the image name and tag
IMAGE_NAME = guojingneo/ai-world
IMAGE_TAG = detection2

GIT_COMMIT=$(shell git rev-parse --short HEAD)

BASE_IMAGE_NAME = guojingneo/ai-base
BASE_IMAGE_TAG = latest

pwd := $(shell pwd)

CUDA_AVAILABLE := $(shell which nvcc)
ifneq ($(CUDA_AVAILABLE),)
	HAS_CUDA = 1
else
	HAS_CUDA = 0
endif

# Define the build command
build:
	docker build -f ./Dockerfile.app -t $(IMAGE_NAME):$(IMAGE_TAG)-$(GIT_COMMIT) .

# Define the clean command
clean:
	docker rmi $(IMAGE_NAME):$(IMAGE_TAG)-$(GIT_COMMIT)

# Define the push command
push:
	docker push $(IMAGE_NAME):$(IMAGE_TAG)-$(GIT_COMMIT)

# Define the run command
run:
	@if [ "$(HAS_CUDA)" -eq "1" ]; then \
		docker run --gpus all -p 7860:7860 -v $(HOME)/.cache/huggingface:$(HOME)/.cache/huggingface -v $(HOME)/.deepface:$(HOME)/.deepface --rm -it --name ai-world $(IMAGE_NAME):$(IMAGE_TAG)-$(GIT_COMMIT); \
	else \
		docker run -p 7860:7860 -v $(HOME)/.cache/huggingface:$(HOME)/.cache/huggingface -v $(HOME)/.deepface:$(HOME)/.deepface --rm -it --name ai-world $(IMAGE_NAME):$(IMAGE_TAG)-$(GIT_COMMIT); \
	fi
# Define the base command
base:
	docker build -f ./deploy/Dockerfile.base -t $(BASE_IMAGE_NAME):$(BASE_IMAGE_TAG) .
	# docker push $(BASE_IMAGE_NAME):$(BASE_IMAGE_TAG)



