from PIL import Image ,ImageDraw, ImageFont
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms as T
import matplotlib.pyplot as plt
import cv2


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


def draw_detect(image_path,boxes, scores,labels, threshold=0.5, label_font=None):
    """
    Draw bounding boxes and labels on the image.

    Args:
        image (PIL.Image): The input image.
        boxes (List[Tuple[float]]): The bounding boxes in format (xmin, ymin, xmax, ymax).
        labels (List[str]): The labels corresponding to the bounding boxes.
        scores (List[float]): The confidence scores for each bounding box.
        threshold (float, optional): The confidence threshold for displaying a bounding box. Default is 0.5.
        label_font (str, optional): The path to a TTF font file for rendering label text. Default is None.
    """
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(label_font, size=16) if label_font else None

    for box, label, score in zip(boxes, labels, scores):
        if score >= threshold:
            draw.rectangle(box, outline='red', width=2)
            text = f"{label}: {score:.2f}"
            text_size = draw.textsize(text, font=font)
            draw.rectangle([box[0], box[1] - text_size[1], box[0] + text_size[0], box[1]], fill='red')
            draw.text((box[0], box[1] - text_size[1]), text, fill='white', font=font)
    image.show()

def draw_instance(image_path,boxes,labels,masks):
    """
    Draw bounding boxes and masks on the image.

    Args:
        image_path (str): The path to the input image.
        boxes (np.ndarray): The bounding boxes in format (xmin, ymin, xmax, ymax).
        labels (np.ndarray): The labels corresponding to the bounding boxes.
        masks (np.ndarray): The segmentation masks corresponding to the bounding boxes.
    """
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    # Create a color map for each class ID
    colors = np.random.uniform(0, 255, size=(len(labels), 3)).astype(np.uint8)
    for i, (box, mask) in enumerate(zip(boxes, masks)):
        color = tuple(colors[i])
        alpha = 0.5
        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        draw_mask = ImageDraw.Draw(overlay)
        draw_mask.polygon(mask.flatten().tolist(), fill=color+(int(alpha*255),))
        image = Image.alpha_composite(image, overlay).convert('RGB')
        draw.rectangle(box, outline=color, width=2)
        draw.text((box[0], box[1]), labels[i], fill=color)
    # Show the image using plt.imshow()
    plt.imshow(image)
    plt.axis('off')
    plt.show()









