from PIL import Image ,ImageDraw, ImageFont
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torchvision.utils as utils
import cv2
import os


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
    # input_image = Image.open(image_path)
    # Load an image using OpenCV
    input_image = cv2.imread(image_path)

    # Convert the image from BGR to RGB color format
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    preprocess = transforms.Compose([
        # transforms.Resize(target_size),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    scale_factor = ()
    # scale_factor = cal_scale_factor((target_size,target_size),input_image.size)

    return input_batch, scale_factor, input_image.shape



def label_to_class(labels, class_dict):
    """
    Converts a list of labels to a list of corresponding classes using a dictionary.
    """
    classes = []
    for label in labels:
        classes.append(class_dict[label])

    return np.array(classes)


def draw_detect(image_path,result):
    image = Image.open(image_path)
    transform = transforms.Compose([transforms.ToTensor()])
    tensor_image = transform(image)
    tensor_image = (tensor_image * 255).to(torch.uint8)
    
    print(tensor_image.shape)
    print(result["classes"].shape)
    
    overlay_images = utils.draw_bounding_boxes(tensor_image,result["boxes"],labels=result["classes"],width=5,colors="red",
                                               fill=False,font=None,font_size=20)
    overlay_images = transforms.ToPILImage()(overlay_images)
    overlay_images.show()

def draw_instance(image_path,result):
    image = Image.open(image_path)
    transform = transforms.Compose([transforms.ToTensor()])
    tensor_image = transform(image)
    tensor_image = (tensor_image * 255).to(torch.uint8)
    print(tensor_image.shape)
    masks = result["masks"].squeeze(1)
    print(masks.shape)
    print(result["classes"].shape)
    # Draw the segmentation masks on top of the images
    overlay_images = utils.draw_segmentation_masks(tensor_image, masks, alpha=0.2)
    overlay_images = utils.draw_bounding_boxes(tensor_image,result["boxes"],labels=result["classes"],width=5,colors="red",
                                               fill=False,font=None,font_size=20)
    overlay_images = transforms.ToPILImage()(overlay_images)
    overlay_images.show()

def draw_masks(image_path,result):
    image = Image.open(image_path)
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(result.shape[0])])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    # plot the semantic segmentation predictions of 21 classes in each color
    r = Image.fromarray(result.byte().cpu().numpy()).resize(image.size)
    r.putpalette(colors)

    plt.imshow(r)








