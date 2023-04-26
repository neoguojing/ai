from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms as T
import matplotlib.pyplot as plt

def load_image_as_numpy(image_path):
    """
    Loads an image from the given file path using the PIL library.
    """
    image = Image.open(image_path)
    array = np.array(image)
    return array

# Define the function for category encoding
def category_encode(labels):
    # Use PyTorch's built-in one-hot encoding function
    encoded_labels = torch.nn.functional.one_hot(labels)
    return encoded_labels


def scale_bbox(bboxs, factor):
    """
    Scales the given list of bounding boxes by the given scale factor.
    """
    print("factor",factor)
    scaled_bboxs = []
    print("origin",bboxs)
    for bbox in bboxs:
        
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        new_width = int(width * factor)
        new_height = int(height * factor)
        new_x1 = int(x1 - (new_width - width) / 2)
        new_y1 = int(y1 - (new_height - height) / 2)
        new_x2 = new_x1 + new_width
        new_y2 = new_y1 + new_height
        scaled_bboxs.append([new_x1, new_y1, new_x2, new_y2])
    print("scaled_bboxs",scaled_bboxs)
    return np.array(scaled_bboxs)




def cal_scale_factor(original_size, target_size):
    """
    Calculates the scale factor needed to resize an image from its original size to the target size.
    """
    original_width, original_height = original_size
    target_width, target_height = target_size
    width_scale_factor = target_width / original_width
    height_scale_factor = target_height / original_height
    return min(width_scale_factor, height_scale_factor)


def image_preprocessor(image_path, target_size=256):
    """
    Preprocesses an image for input into a machine learning model.
    """

    input_image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(target_size),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    scale_factor = cal_scale_factor((target_size,target_size),input_image.size)

    return input_batch, scale_factor


def label_to_class(labels, class_dict):
    """
    Converts a list of labels to a list of corresponding classes using a dictionary.
    """
    classes = []
    for label in labels:
        classes.append(class_dict[label])

    return np.array(classes)


def draw_detect(image_path,boxes, scores, labels,threshhold=0.5):
    image = Image.open(image_path)
    image = T.ToTensor()(image)
    image = image.permute(1,2,0).numpy()
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for box, score, label in zip(boxes, scores, labels):
        if score < threshhold:
            continue  # Skip low-confidence detections
            
        left = box[0] 
        top = box[1] 
        width = box[2] - left 
        height = box[3] - top
        
        rect = plt.Rectangle((left, top), width, height, fill=False, color='red', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(left, top - 5, f'{label} {score:.2f}', fontsize=8, color='green')

    plt.show()

def draw_instance(image_path,boxes,labels,masks):
    image = Image.open(image_path)
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)

    for i in range(len(boxes)):
        mask = masks[i, 0]
        x1, y1, x2, y2 = boxes[i]
        width = x2 - x1
        height = y2 - y1
        ax.imshow(mask, alpha=0.5, extent=[x1, x1+width, y1, y1+height], cmap='Reds')
        label = labels[i]
        ax.text(x1, y1, f"{label}", fontsize=12, color='white', bbox=dict(facecolor='red', alpha=0.5, pad=0), 
            verticalalignment='top')

    plt.show()






