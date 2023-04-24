import urllib.request
import os
import json
import pickle

DATASET_PREFIX = os.environ.get('DATASET_PREFIX', '')
IMAGENET_LABELS_FILE = DATASET_PREFIX + "imagenet_class_index.json"
CIFAR100_LABELS_FILE = DATASET_PREFIX + "cifar100_labels.txt"
CIFAR10_LABELS_FILE = DATASET_PREFIX + "cifar10_labels.meta"
PASCAL_VOC_LABELS_FILE = DATASET_PREFIX + "pascal_voc_labels.txt"
PLACES365_LABELS_FILE = DATASET_PREFIX + "categories_places365.txt"
COCO_LABELS_FILE = DATASET_PREFIX + "coco_labels.txt"


def get_imagenet_labels():
    # Download the labels file from the internet
    
    if not os.path.exists(IMAGENET_LABELS_FILE):
        url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
        urllib.request.urlretrieve(url, IMAGENET_LABELS_FILE)
    
    # Load the labels file
  
    with open(IMAGENET_LABELS_FILE) as f:
        class_idx = json.load(f)
    
    # Extract the labels
    labels = [class_idx[str(k)][1] for k in range(len(class_idx))]
    return labels

# Call the function to get the labels from ImageNet
imagenet_labels = get_imagenet_labels()


def get_cifar100_labels():
    # Download the labels file from the internet
    if not os.path.exists(CIFAR100_LABELS_FILE):
        url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
        urllib.request.urlretrieve(url, "cifar-100-python.tar.gz")
        os.system("tar -xzf cifar-100-python.tar.gz cifar-100-python/meta")
        os.system("mv cifar-100-python/meta " + CIFAR100_LABELS_FILE)
    
    # Load the labels file
    with open(CIFAR100_LABELS_FILE, "r") as f:
        cifar100_labels = f.readlines()
        cifar100_labels = [label.strip() for label in cifar100_labels]
    
    # Return the CIFAR-100 labels
    return cifar100_labels

# Call the function to get the labels from CIFAR-100
cifar100_labels = get_cifar100_labels()

def get_cifar10_labels():
    # Download the labels file from the internet
    if not os.path.exists(CIFAR10_LABELS_FILE):
        url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        urllib.request.urlretrieve(url, "cifar-10-python.tar.gz")
        os.system("tar -xzf cifar-10-python.tar.gz")
    
    # Load the labels file
    with open(CIFAR10_LABELS_FILE, "rb") as f:
        cifar10_labels = pickle.load(f, encoding='bytes')
        cifar10_labels = [label.decode('utf-8') for label in cifar10_labels[b'label_names']]
    
    # Return the CIFAR-10 labels
    return cifar10_labels

# Call the function to get the labels from CIFAR-10
cifar10_labels = get_cifar10_labels()


def get_pascal_voc_labels():
    # Download the labels file from the internet
    if not os.path.exists(PASCAL_VOC_LABELS_FILE):
        url = "https://raw.githubusercontent.com/amikelive/coco-labels/master/pascal-voc.names"
        urllib.request.urlretrieve(url, PASCAL_VOC_LABELS_FILE)
    
    # Load the labels file
    with open(PASCAL_VOC_LABELS_FILE, "r") as f:
        pascal_voc_labels = f.readlines()
        pascal_voc_labels = [label.strip() for label in pascal_voc_labels]
    
    # Return the Pascal VOC labels
    return pascal_voc_labels

# Call the function to get the labels from Pascal VOC
pascal_voc_labels = get_pascal_voc_labels()


def get_places365_labels():
    # Download the labels file from the internet
    if not os.path.exists(PLACES365_LABELS_FILE):
        url = "http://places2.csail.mit.edu/models_places365/categories_places365.txt"
        urllib.request.urlretrieve(url, PLACES365_LABELS_FILE)
    
    # Load the labels file
    with open(PLACES365_LABELS_FILE, "r") as f:
        places365_labels = f.readlines()
        places365_labels = [label.strip().split(' ')[0][3:] for label in places365_labels]
    
    # Return the Places365 labels
    return places365_labels

# Call the function to get the labels from Places365
places365_labels = get_places365_labels()

def get_coco_labels():
    # Download the labels file from the internet
    if not os.path.exists(COCO_LABELS_FILE):
        labels_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
        urllib.request.urlretrieve(labels_url, COCO_LABELS_FILE)
    
    # Load the labels file
    with open(COCO_LABELS_FILE, "r") as f:
        coco_labels = f.readlines()
        coco_labels = [label.strip() for label in coco_labels]
    
    # Return the COCO labels
    return coco_labels

# Call the function to get the labels from COCO
coco_labels = get_coco_labels()
