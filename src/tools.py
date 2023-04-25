from PIL import Image
import numpy as np
import torch

def load_image(image_path):
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


def scale_bbox(bbox, scale_factor):
    """
    Scales the given bounding box by the given scale factor.
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    new_x1 = int(x1 - (new_width - width) / 2)
    new_y1 = int(y1 - (new_height - height) / 2)
    new_x2 = new_x1 + new_width
    new_y2 = new_y1 + new_height
    return np.array([new_x1, new_y1, new_x2, new_y2])



def cal_scale_factor(original_size, target_size):
    """
    Calculates the scale factor needed to resize an image from its original size to the target size.
    """
    original_width, original_height = original_size
    target_width, target_height = target_size
    width_scale_factor = target_width / original_width
    height_scale_factor = target_height / original_height
    return min(width_scale_factor, height_scale_factor)


