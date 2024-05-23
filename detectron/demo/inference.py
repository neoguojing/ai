
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
        
        self.instance_segment_cfg = ModelConfig(ModelCategory.INSTANCE_SEGMENTATION, 
                                         model_path="COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
                                         cfg_path="../configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml").get_cfg()
        self.panoptic_segment_cfg = ModelConfig(ModelCategory.INSTANCE_SEGMENTATION, 
                                         model_path="COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml",
                                         cfg_path="../configs/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml").get_cfg()
        self.keypoints_cfg = ModelConfig(ModelCategory.KEYPOINTS, 
                                         model_path="COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml",
                                         cfg_path="../configs/COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml").get_cfg()
        
        self.aug = T.ResizeShortestEdge(
            [256, 256], 256
        )
    
    def image_processor(self,original_image):
        
        # if self.input_format == "RGB":
        #         # whether the model expects BGR inputs or RGB
        #         original_image = original_image[:, :, ::-1]
        height, width = original_image.shape[:2]
        image = self.aug.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        image.to("cpu")

        inputs = {"image": image, "height": height, "width": width}
        return inputs

    def extract(self, image_path: str="./test.png"):
        """
        Perform classification on an image using Detectron2.
        """
        
        model = TorchModelFactory.create_feature_extract_model("resnet")

        from detectron2.data.detection_utils import read_image
    
        img = read_image(image_path)
        input = self.image_processor(img)

        with torch.no_grad():
            input_batch = input["image"].unsqueeze(0)
            output = model(input_batch)
            
        result = {"features":output}
        print(result)
        return result
    
    def classify(self, image_path: str="./cat.jpg"):
        """
        Perform classification on an image using Detectron2.
        """
        model = TorchModelFactory.create_feature_extract_model("resnet")

        from detectron2.data.detection_utils import read_image
    
        img = read_image(image_path)
        input = self.image_processor(img)

        with torch.no_grad():
            input_batch = input["image"].unsqueeze(0)
            output = model(input_batch)
        
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        print(probabilities,probabilities.shape)
        top5_prob, top5_catid = torch.topk(probabilities, 1)
        print(top5_prob,top5_catid)
        result = {"features":output,"pred_classes":top5_catid,"scores":top5_prob}
        return result
    
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
        model = TorchModelFactory.create_semantic_model("deeplabv3")
        return model
    
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
    # f.semantic_segment()
    f.classify()

    