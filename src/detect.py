# Import necessary libraries
import torch
from torchvision.models import detection
import numpy as np

def load_image(image_path):
    """
    Loads an image from the given file path using the PIL library.
    """
    image = Image.open(image_path)
    return image

# Define model factory
def model_factory(model_name):
    if model_name == 'RetinaNet':
        # [{'boxes': tensor([], size=(0, 4)), 'scores': tensor([]), 'labels': tensor([], dtype=torch.int64)}]
        # Load RetinaNet model in inference mode
        model = detection.retinanet_resnet50_fpn(pretrained=True, pretrained_backbone=True)
    elif model_name == 'FasterRCNN':
        # [{'boxes': tensor([[  0.,   0., 640., 669.],
        # [  0.,   0., 640., 669.]], device='cuda:0'), 'labels': tensor([67,  1], device='cuda:0'), 
        # 'scores': tensor([1., 1.], device='cuda:0')}]
        # Load FasterRCNN model in inference mode
        model = detection.fasterrcnn_resnet50_fpn(pretrained=True, pretrained_backbone=True)
    elif model_name == 'SSDLite':
        # Load SSD Lite model in inference mode
        model = detection.ssd_lite_mobilenet_v3_large(pretrained=True, pretrained_backbone=True)
    elif model_name == 'Yolov5':
        # Load Yolov5 model in inference mode
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        
    else:
        raise ValueError('Invalid model name')
    model.eval()
    return model


# Define function to detect with a given model
def detect_with_model(image, model_name):
    # Get model
    model = model_factory(model_name)
    # Preprocess image
    image = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0)

    if torch.cuda.is_available():
        image = image.cuda()
        model.eval().cuda()
        print("using gpu")

    # Perform detection
    with torch.no_grad():
        detections = model(image)
    # detections = model.postprocess(detections)
    return detections



def post_process_detections(outputs, confidence_threshold=0.5, nms_iou_threshold=0.5):
    """
    Post-process RetinaNet detections to obtain final object detections.
    Args:
        outputs (list): List of outputs from RetinaNet model, including classification logits,
                        localization logits, and anchor boxes.
        confidence_threshold (float): Confidence threshold for objectness score.
        nms_iou_threshold (float): IOU threshold for non-maximum suppression (NMS).
    Returns:
        detections (list): List of final object detections, each containing class ID, confidence score,
                            and bounding box coordinates.
    """
    classification, regression, anchors = outputs
    num_anchors = anchors.shape[0]
    num_classes = classification.shape[2] - 1
    
    # Apply sigmoid activation to objectness scores
    objectness = torch.sigmoid(classification[:, :, 0])
    
    # Apply softmax activation to class scores
    class_probs = torch.softmax(classification[:, :, 1:], dim=2)
    
    # Apply regression offsets to anchor boxes
    bbox_deltas = regression.reshape((-1, 4))
    anchors = anchors.repeat(classification.shape[0], 1, 1).reshape((-1, 4))
    pred_boxes = bbox_transform_inv(anchors, bbox_deltas)
    
    # Filter out boxes with low objectness score
    keep = objectness > confidence_threshold
    objectness = objectness[keep]
    class_probs = class_probs[keep, :]
    pred_boxes = pred_boxes[keep, :]
    
    # Apply non-maximum suppression (NMS)
    class_scores, class_ids = torch.max(class_probs, dim=1)
    class_ids += 1  # Add 1 to class IDs to account for background class at index 0
    detections = []
    for class_id in range(1, num_classes+1):
        keep = class_ids == class_id
        if keep.any():
            class_scores_filtered = class_scores[keep]
            pred_boxes_filtered = pred_boxes[keep, :]
            keep_indices = nms(pred_boxes_filtered, class_scores_filtered, nms_iou_threshold)
            class_scores_filtered = class_scores_filtered[keep_indices]
            pred_boxes_filtered = pred_boxes_filtered[keep_indices, :]
            for score, box in zip(class_scores_filtered, pred_boxes_filtered):
                detections.append((class_id, float(score), [float(coord) for coord in box]))
    
    return detections

def bbox_transform_inv(boxes, deltas):
    """
    Apply regression offsets to anchor boxes to obtain predicted bounding boxes.
    Args:
        boxes (torch.Tensor): Tensor of shape (N, 4) containing anchor boxes.
        deltas (torch.Tensor): Tensor of shape (N, 4) containing regression offsets.
    Returns:
        pred_boxes (torch.Tensor): Tensor of shape (N, 4) containing predicted bounding boxes.
    """
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights
    
    dx = deltas[:, 0]
    dy = deltas[:, 1]
    dw = deltas[:, 2]
    dh = deltas[:, 3]
    
    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights
    
    pred_boxes = torch.zeros_like(deltas)
    pred_boxes[:, 0] = pred_ctr_x - 0.5 * pred_w
    pred_boxes[:, 1] = pred_ctr_y - 0.5 * pred_h
    pred_boxes[:, 2] = pred_ctr_x + 0.5 * pred_w
    pred_boxes[:, 3] = pred_ctr_y + 0.5 * pred_h
    
    return pred_boxes

def nms(boxes, scores, iou_threshold):
    """
    Apply non-maximum suppression (NMS) to filter out overlapping boxes.
    Args:
        boxes (torch.Tensor): Tensor of shape (N, 4) containing predicted bounding boxes.
        scores (torch.Tensor): Tensor of shape (N,) containing class scores.
        iou_threshold (float): IOU threshold for NMS.
    Returns:
        keep_indices (list): List of indices to keep after NMS.
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort(descending=True)
    
    keep_indices = []
    while order.numel() > 0:
        i = order[0]
        keep_indices.append(i)
        if order.numel() == 1:
            break
        
        xx1 = torch.max(x1[i], x1[order[1:]])
        yy1 = torch.max(y1[i], y1[order[1:]])
        xx2 = torch.min(x2[i], x2[order[1:]])
        yy2 = torch.min(y2[i], y2[order[1:]])
        w = torch.max(torch.zeros_like(xx2), xx2 - xx1 + 1)
        h = torch.max(torch.zeros_like(yy2), yy2 - yy1 + 1)
        intersection = w * h
        iou = intersection / (areas[i] + areas[order[1:]] - intersection)
        indices_to_keep = (iou <= iou_threshold).nonzero().squeeze()
        if indices_to_keep.numel() == 0:
            break
        order = order[indices_to_keep + 1]
    
    return keep_indices
