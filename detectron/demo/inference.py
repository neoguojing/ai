
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
    KEYPOINTS = "keypoints"
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
        if cfg_path is not None:
            self.cfg.merge_from_file(cfg_path)
        
        if model_path is not None:
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_path)
        self.thresh_hold = thresh_hold

        if model_type == ModelCategory.OBJECT_DETECTION:
            pass
        elif model_type == ModelCategory.IMAGE_FEATURE_EXTRACT:
            self.cfg.MODEL.RESNETS.OUT_FEATURES = ["res2", "res3", "res4", "res5"] 

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
        # TODO
        self.extract_cfg = ModelConfig(ModelCategory.IMAGE_FEATURE_EXTRACT, 
                                         model_path="COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
                                         cfg_path="../configs/COCO-Detection/retinanet_R_101_FPN_3x.yaml").get_cfg()
        
        self.classification_cfg = ModelConfig(ModelCategory.IMAGE_CLASSIFICATION, 
                                         model_path="COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
                                         cfg_path="../configs/COCO-Detection/retinanet_R_101_FPN_3x.yaml").get_cfg()
        
        self.semantic_segment_cfg = ModelConfig(ModelCategory.SEMANTIC_SEGMENTATION, 
                                         model_path="Misc/semantic_R_50_FPN_1x.yaml",
                                         cfg_path="../configs/Misc/semantic_R_50_FPN_1x.yaml").get_cfg()
        
        self.instance_segment_cfg = ModelConfig(ModelCategory.INSTANCE_SEGMENTATION, 
                                         model_path="COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
                                         cfg_path="../configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml").get_cfg()
        self.panoptic_segment_cfg = ModelConfig(ModelCategory.INSTANCE_SEGMENTATION, 
                                         model_path="COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml",
                                         cfg_path="../configs/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml").get_cfg()
        self.keypoints_cfg = ModelConfig(ModelCategory.KEYPOINTS, 
                                         model_path="COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml",
                                         cfg_path="../configs/COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml").get_cfg()
        
    def extract(self, image_path: str="./test.png"):
        """
        Perform classification on an image using Detectron2.
        """

        outputs,_ = p.run_on_image(img)
       
        return outputs
    
    def classify(self, image_path: str):
        """
        Perform classification on an image using Detectron2.
        """
        import torch
        num_classes = 1000  # 示例：ImageNet 数据集的类别数
        model.fc = torch.nn.Linear(num_features, num_classes)  # 替换全连接层
        return 
    
    def onstep_detect(self, image_path: str= "./test.png", confidence_threshold: float = 0.5):
        """
        Perform on step object detection on an image using Detectron2.
        """
        p = InferenceBase(self.onstep_detection_cfg,thresh_hold=confidence_threshold)
        print(self.onstep_detection_cfg)
        img = p.read_image(image_path)
        outputs,vis_output = p.run_on_image(img)
        if vis_output is not None:
            p.save_vis_image(vis_output)
       
        return outputs
    
    def detect(self, image_path: str = "./test.png", confidence_threshold: float = 0.5):
        """
        Perform object detection on an image using Detectron2.
        """
        p = InferenceBase(self.detection_cfg,thresh_hold=confidence_threshold)
        print(self.detection_cfg)
        img = p.read_image(image_path)
        outputs,vis_output = p.run_on_image(img)
        if vis_output is not None:
            p.save_vis_image(vis_output)
        print(outputs)
       
        return outputs
    
    def instance_segment(self, image_path: str="./test.png"):
        """
        Perform instance segmentation on an image using Detectron2.
        """
        p = InferenceBase(self.instance_segment_cfg)
        img = p.read_image(image_path)
        outputs,vis_output = p.run_on_image(img)
        if vis_output is not None:
            p.save_vis_image(vis_output)
        print(outputs)
       
        return outputs
        
    def semantic_segment(self, image_path: str="./test.png"):
        """
        Perform instance segmentation on an image using Detectron2.
        """
        p = InferenceBase(self.semantic_segment_cfg)
        img = p.read_image(image_path)
        outputs,vis_output = p.run_on_image(img)
        if vis_output is not None:
            p.save_vis_image(vis_output)
        print(outputs)
        return outputs
    
    def panoptic_segment(self, image_path: str="./test.png"):
        """
        Perform panoptic segmentation on an image using Detectron2.
        """
        p = InferenceBase(self.panoptic_segment_cfg)
        img = p.read_image(image_path)
        outputs,vis_output = p.run_on_image(img)
        if vis_output is not None:
            p.save_vis_image(vis_output)
        print(outputs)
        return outputs
        
    def keypoint(self, image_path: str="./test.png"):
        """
        Perform keypoint on an image using Detectron2.
        """
        p = InferenceBase(self.keypoints_cfg)
        print(self.detection_cfg)
        img = p.read_image(image_path)
        outputs,vis_output = p.run_on_image(img)
        if vis_output is not None:
            p.save_vis_image(vis_output)
        print(outputs)
       
        return outputs
        
if __name__ == "__main__":
    f = ModelFactory()
    f.semantic_segment()

    