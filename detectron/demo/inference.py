
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
import detectron2.data.transforms as T
from predictor import InferenceBase
from model_factory import TorchModelFactory
import torch
import torchvision.transforms as transforms
from PIL import Image

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
    def __init__(self,model_type, model_path: str=None,cfg_path: str= None,thresh_hold: float = 0.5):
        self.cfg = get_cfg()
        if cfg_path is not None:
            self.cfg.merge_from_file(cfg_path)
        
        if model_path is not None:
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_path)
        self.thresh_hold = thresh_hold

        
        if model_type == ModelCategory.IMAGE_FEATURE_EXTRACT:
            self.cfg.TASK_TYPE = "feature"
            self.cfg.MODEL.WEIGHTS = None
        elif model_type == ModelCategory.IMAGE_CLASSIFICATION:
            self.cfg.TASK_TYPE = "classfication"
            self.cfg.MODEL.WEIGHTS = None
        elif model_type == ModelCategory.SEMANTIC_SEGMENTATION:
            self.cfg.TASK_TYPE = "semantic"
            self.cfg.MODEL.WEIGHTS = None
        # elif model_type == ModelCategory.OBJECT_DETECTION:
        #     self.cfg.TASK_TYPE = "detect"
        #     self.cfg.MODEL.WEIGHTS = None

    def get_cfg(self,):
        return self.cfg

class ModelFactory:
    def __init__(self):
        self.detection_cfg = ModelConfig(ModelCategory.OBJECT_DETECTION, 
                                         model_path="COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
                                         cfg_path="../configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml").get_cfg()
        
        self.feature_cfg = ModelConfig(ModelCategory.IMAGE_FEATURE_EXTRACT, 
                                         model_path=None,
                                         cfg_path=None).get_cfg()
        
        self.classification_cfg = ModelConfig(ModelCategory.IMAGE_CLASSIFICATION, 
                                         model_path=None,
                                         cfg_path=None).get_cfg()
        
        self.onstep_detection_cfg = ModelConfig(ModelCategory.ONE_STEP_OBJECT_DETECTION, 
                                         model_path="COCO-Detection/retinanet_R_101_FPN_3x.yaml",
                                         cfg_path="../configs/COCO-Detection/retinanet_R_101_FPN_3x.yaml").get_cfg()
        
        self.instance_segment_cfg = ModelConfig(ModelCategory.INSTANCE_SEGMENTATION, 
                                         model_path="COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
                                         cfg_path="../configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml").get_cfg()
        self.semantic_segment_cfg = ModelConfig(ModelCategory.SEMANTIC_SEGMENTATION, 
                                         model_path=None,
                                         cfg_path="../configs/PascalVOC-Detection/faster_rcnn_R_50_FPN.yaml").get_cfg()
        
        self.panoptic_segment_cfg = ModelConfig(ModelCategory.INSTANCE_SEGMENTATION, 
                                         model_path="COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml",
                                         cfg_path="../configs/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml").get_cfg()
        self.keypoints_cfg = ModelConfig(ModelCategory.KEYPOINTS, 
                                         model_path="COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml",
                                         cfg_path="../configs/COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml").get_cfg()
        
        self.aug = T.ResizeShortestEdge(
            [224, 224], 224
        )
    
    def image_processor(self,image_path,resize=256,crop=224):
        from PIL import Image
        input_image = Image.open(image_path).convert('RGB')
        preprocess = transforms.Compose([
            transforms.Resize(resize) if resize is not None else lambda x: x,
            transforms.CenterCrop(crop) if crop is not None else lambda x: x,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)
        return input_batch

    def prepare_meta(self,dataset="voc"):
        print("--------------",self.semantic_segment_cfg.MODEL.WEIGHTS,self.semantic_segment_cfg.DATASETS.TEST)
        meta = None
        if dataset == "voc":
            meta = MetadataCatalog.get("voc_2007_test")
            meta.stuff_classes = meta.thing_classes
            meta.stuff_classes.insert(0, "background")

        return meta
    def extract(self, image_path: str="./test.png"):
        """
        Perform classification on an image using Detectron2.
        """    
        p = InferenceBase(self.feature_cfg,thresh_hold=0.5)
        print(self.feature_cfg.MODEL.WEIGHTS,self.feature_cfg.TASK_TYPE)
        # img = p.read_image(image_path)
        
        img = Image.open(image_path).convert('RGB')
        outputs,vis_output = p.run_on_image(img)
        if vis_output is not None:
            p.save_vis_image(vis_output)
       
        return outputs
    
    def classify(self, image_path: str="./cat.jpg"):
        """
        Perform classification on an image using Detectron2.
        """

        p = InferenceBase(self.classification_cfg,thresh_hold=0.5)
        print(self.classification_cfg.MODEL.WEIGHTS,self.classification_cfg.TASK_TYPE)
        # img = p.read_image(image_path)
        
        img = Image.open(image_path).convert('RGB')
        outputs,vis_output = p.run_on_image(img)
        if vis_output is not None:
            p.save_vis_image(vis_output)
       
        return outputs
        
    
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
    
        p = InferenceBase(self.semantic_segment_cfg,thresh_hold=0.5)
        print(self.semantic_segment_cfg.DATASETS.TEST[0])
        # img = p.read_image(image_path)
        img = Image.open(image_path).convert('RGB')
        outputs,vis_output = p.run_on_image(img)
        if vis_output is not None:
            p.save_vis_image(vis_output)
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
    # f.prepare_meta()
    out = f.detect()
    print(out)

    