
# import some common libraries
import numpy as np
import os, json, cv2
import sys
sys.path.append("..")
# import some common detectron2 utilities
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import ColorMode
from predictor import InferenceBase

# 定义模型类别的常量
class ModelCategory:
    IMAGE_FEATURE_EXTRACT = "image_feature_extract"
    IMAGE_CLASSIFICATION = "image_classification"
    OBJECT_DETECTION = "object_detection"
    ONE_STEP_OBJECT_DETECTION = "onestep_object_detection"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    INSTANCE_SEGMENTATION = "instance_segmentation"
    PANOPTIC_SEGMENTATION = "panoptic_segmentation"
    REGRESSION = "regression"
    TEXT_CLASSIFICATION = "text_classification"
    LANGUAGE_MODELLING = "language_modelling"
    TRANSLATION = "translation"
    QA_SYSTEM = "qa_system"
    RECOMMENDATION_SYSTEM = "recommendation_system"
    GENERATIVE_MODELLING = "generative_modelling"
    CONTROL = "control"
    ROBOTICS = "robotics"
    OTHERS = "others"

class ModelConfig:
    cfg: None
    def __init__(self,model_type, model_path: str="",cfg_path: str= "",thresh_hold: float = 0.5):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(cfg_path)
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_path)
        self.thresh_hold = thresh_hold

    def get_cfg(self,):
        return self.cfg

class ModelFactory:
    def __init__(self):
        self.detection_cfg = ModelConfig(ModelCategory.OBJECT_DETECTION, 
                                         model_path="COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
                                         cfg_path="../configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml").get_cfg()
        self.onstep_detection_cfg = ModelConfig(ModelCategory.ONE_STEP_OBJECT_DETECTION, 
                                         model_path="COCO-Detection/retinanet_R_101_FPN_3x.yaml",
                                         cfg_path="../configs/COCO-Detection/retinanet_R_101_FPN_3x.yaml").get_cfg()
        
    def extract(self, image_path: str):
        """
        Perform classification on an image using Detectron2.
        """
        im = cv2.imread(image_path)
        outputs = self.predictor(im)
        class_names = outputs["pred_classes"].cpu().numpy().astype(str)
        scores = outputs["scores"].cpu().numpy()
        return dict(zip(class_names, scores))
    
    def classify(self, image_path: str):
        """
        Perform classification on an image using Detectron2.
        """
        im = cv2.imread(image_path)
        outputs = self.predictor(im)
        class_names = outputs["pred_classes"].cpu().numpy().astype(str)
        scores = outputs["scores"].cpu().numpy()
        return dict(zip(class_names, scores))
    
    def onstep_detect(self, image_path: str= "./test.png", confidence_threshold: float = 0.5):
        """
        Perform on step object detection on an image using Detectron2.
        """
        p = InferenceBase(self.onstep_detection_cfg,thresh_hold=confidence_threshold)
        img = p.read_image(image_path)
        outputs,vis_output = p.run_on_image(img)
        if vis_output is not None:
            p.save_vis_image(vis_output)
        print(outputs)
       
        return outputs
    
    def detect(self, image_path: str = "./test.png", confidence_threshold: float = 0.5):
        """
        Perform object detection on an image using Detectron2.
        """
        p = InferenceBase(self.detection_cfg,thresh_hold=confidence_threshold)
        img = p.read_image(image_path)
        outputs,vis_output = p.run_on_image(img)
        if vis_output is not None:
            p.save_vis_image(vis_output)
        print(outputs)
       
        return outputs
    
    def instance_segment(self, image_path: str, output_type: str = "image"):
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
        
    def semantic_segment(self, image_path: str, output_type: str = "image"):
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
        
if __name__ == "__main__":
    f = ModelFactory()
    f.onstep_detect()
    