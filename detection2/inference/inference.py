import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import ColorMode

class Detectron2Inference:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.predictor = DefaultPredictor(self.model_path)
        self.models = {
            "detection": [
                "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
                "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
                "COCO-Detection/retinanet_R_50_FPN_3x.yaml",
            ],
            "segmentation": [
                "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
                "LVIS-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml",
            ],
            "panoptic": [
                "COCO-PanopticSegmentation/panoptic_deeplab_R_101_FPN_3x.yaml",
                "COCO-PanopticSegmentation/panoptic_deeplab_R_50_FPN_3x.yaml",
            ],
        }
        
    def classify(self, image_path: str):
        """
        Perform classification on an image using Detectron2.
        """
        im = cv2.imread(image_path)
        outputs = self.predictor(im)
        class_names = outputs["pred_classes"].cpu().numpy().astype(str)
        scores = outputs["scores"].cpu().numpy()
        return dict(zip(class_names, scores))
    
    def detect(self, image_path: str, confidence_threshold: float = 0.5):
        """
        Perform object detection on an image using Detectron2.
        """
        im = cv2.imread(image_path)
        outputs = self.predictor(im)
        boxes = outputs["boxes"].cpu().numpy()
        scores = outputs["scores"].cpu().numpy()
        labels = outputs["labels"].cpu().numpy().astype(int)
        return [(box, score, label) for box, score, label in zip(boxes, scores, labels) if score >= confidence_threshold]
    
    def segment(self, image_path: str, output_type: str = "image"):
        """
        Perform instance segmentation on an image using Detectron2.
        """
        im = cv2.imread(image_path)
        outputs = self.predictor(im)
        masks = outputs["masks"].cpu().numpy()
        classes = outputs["classes"].cpu().numpy().astype(int)
        scores = outputs["scores"].cpu().numpy()
        if output_type == "image":
            v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(self.model_path), scale=1.2)
            v = v.draw_instance_predictions(predictions=outputs)
            return v.get_image()[:, :, ::-1]
        else:
            return list(zip(masks, classes, scores))
    
    def panoptic_segment(self, image_path: str, output_type: str = "image"):
        """
        Perform panoptic segmentation on an image using Detectron2.
        """
        im = cv2.imread(image_path)
        outputs = self.predictor(im)
        panoptic_seg = outputs["panoptic_seg"].cpu().numpy()
        if output_type == "image":
            v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(self.model_path), scale=1.2)
            v = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), outputs["segments_info"])
            return v.get_image()[:, :, ::-1]
        else:
            return panoptic_seg
