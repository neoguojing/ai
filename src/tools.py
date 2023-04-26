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

    input_image = Image.open(image_path)
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

    return input_batch, scale_factor


def label_to_class(labels, class_dict):
    """
    Converts a list of labels to a list of corresponding classes using a dictionary.
    """
    classes = []
    for label in labels:
        classes.append(class_dict[label])

    return np.array(classes)


def draw_detect(image_path,boxes, labels, scores, threshold=0.5, label_font=None):
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
    return image

def draw_instance(image_path,boxes,labels,masks):
    image = Image.open(image_path)

    # Convert the masks to NumPy arrays and resize them to the size of the input image
    masks = [cv2.resize(mask, image.shape[1::-1]) for mask in masks]
    # his means that any pixel with a value greater than 0.5 (i.e., any pixel that belongs to the object with a 
    # confidence score of at least 0.5) is set to 1, and any pixel with a value less than or equal to 0.5
    #  (i.e., any pixel that does not belong to the object with a confidence score less than 0.5) is set to 0.
    masks = np.array(masks) > 0.5
    
    # Create a color map for each class ID
    colors = np.random.uniform(0, 255, size=(len(labels), 3)).astype(np.uint8)
    for i, (box, mask) in enumerate(zip(boxes, masks)):
        color = colors[i]
        alpha = 0.5
        overlay = np.zeros_like(image)
        overlay = cv2.fillPoly(overlay, [mask.astype(np.int32)], color.tolist())
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        cv2.rectangle(image, tuple(box[:2]), tuple(box[2:]), color.tolist(), 2)
    # Show the image using plt.imshow()
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    # fig, ax = plt.subplots(1, figsize=(10, 10))
    # ax.imshow(image)

    # for i in range(len(boxes)):
    #     mask = masks[i, 0]
    #     x1, y1, x2, y2 = boxes[i]
    #     width = x2 - x1
    #     height = y2 - y1
    #     ax.imshow(mask, alpha=0.5, extent=[x1, x1+width, y1, y1+height], cmap='Reds')
    #     label = labels[i]
    #     ax.text(x1, y1, f"{label}", fontsize=12, color='white', bbox=dict(facecolor='red', alpha=0.5, pad=0), 
    #         verticalalignment='top')

    # plt.show()








