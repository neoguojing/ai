
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
from detectron2.data.detection_utils import pil_image_handler

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
        
        self.need_save_images = False
    
    def serialize(self,output):
        serialized = None
        # print(output)
        if "instances" in output:
            serialized = {
                'image_height': output["instances"].image_size[0],
                'image_width': output["instances"].image_size[1],
                'pred_boxes': output["instances"].pred_boxes.tensor.tolist(),
                'scores': output["instances"].scores.tolist(),
                'pred_classes': output["instances"].pred_classes.tolist()
            }

            if hasattr(output["instances"],"pred_masks"):
                # serialized["pred_masks"] = output["instances"].pred_masks.tolist()
                print("instances.pred_masks",output["instances"].pred_masks.shape)
            if hasattr(output["instances"],"pred_keypoints"):
                serialized["pred_keypoints"] = output["instances"].pred_keypoints.tolist()
        if "sem_seg" in output:
            # serialized["sem_seg"] = output["sem_seg"].tolist()
            print("sem_seg:",output["sem_seg"].shape)
        
        if "panoptic_seg" in output:
            print("panoptic_seg:",output["panoptic_seg"][0].shape)
            # print("panoptic_seg:",output["panoptic_seg"])
            serialized["panoptic_seg"] = output["panoptic_seg"][1]
        if "sem_segs" in output:
            print("sem_segs:",output["sem_segs"].shape)
        
        if "classfication" in output:
            serialized = []
            for item in output["classfication"]:
                print("classfication:",item["feature"].shape)
                row = {
                    # "feature": item["feature"].tolist(),
                    "score": item["score"].tolist(),
                    "pred_class": item["pred_class"].tolist(),
                }
                serialized.append(row)

        if "features" in output:
            print("features:",output["features"].shape)
            serialized = {
                "features":output["features"].tolist(),
            }

        return serialized
    
    def predict(self,pil_image,task_type="panoptic"):
        result = None
        vis_output = None
        if task_type == "panoptic":
            result,vis_output = self.panoptic_segment(input_image=pil_image)
        elif task_type == "detect":
            result,vis_output = self.detect(input_image=pil_image)
        elif task_type == "classification":
            result = self.classify(input_image=pil_image)
        elif task_type == "instance":
            result,vis_output = self.instance_segment(input_image=pil_image)
        elif task_type == "semantic":
            result,vis_output = self.semantic_segment(input_image=pil_image)
        elif task_type == "feature":
            result = self.extract(input_image=pil_image)
        elif task_type == "keypoint":
            result,vis_output = self.keypoint(input_image=pil_image)
        elif task_type == "onestep_detect":
            result,vis_output = self.onstep_detect(input_image=pil_image)

        pil_images = []
        if vis_output is not None:
            if self.need_save_images:
                self.save_vis_image(vis_output)
            
            pil_images = self.visimage_to_pil(vis_output)
        
        return self.serialize(result),pil_images

    def save_vis_image(self,visimages):
        import uuid
        for visimage in visimages:
            unique_id = uuid.uuid1()
            visualized_image = visimage.get_image()[:, :, ::-1]
            cv2.imwrite(self.output_dir+str(unique_id)+".png", visualized_image)

    def visimage_to_pil(self,visimages):
        pil_images = []
        for visimage in visimages:
            visualized_image = visimage.get_image()[:, :, ::-1]
            pil_image = Image.fromarray(visualized_image)
            pil_images.append(pil_image)
        return pil_images


    def extract(self, input_image=None,image_path: str="./test.png"):
        """
        Perform classification on an image using Detectron2.
        """    
        p = InferenceBase(self.feature_cfg,thresh_hold=0.5)
        print(self.feature_cfg.MODEL.WEIGHTS,self.feature_cfg.TASK_TYPE)
        # img = p.read_image(image_path)
        if input_image is None and image_path is not None:
            input_image = Image.open(image_path).convert('RGB')
            input_image = pil_image_handler(input_image)
        outputs,_ = p.run_on_image(input_image)
       
        return outputs
    
    def classify(self, input_image=None,image_path: str="./cat.jpg"):
        """
        Perform classification on an image using Detectron2.
        """

        p = InferenceBase(self.classification_cfg,thresh_hold=0.5)
        print(self.classification_cfg.MODEL.WEIGHTS,self.classification_cfg.TASK_TYPE)
        # img = p.read_image(image_path)
        if input_image is None and image_path is not None:
            input_image = Image.open(image_path).convert('RGB')
            input_image = pil_image_handler(input_image)
        outputs,_ = p.run_on_image(input_image)
        
        return outputs
        
    
    def onstep_detect(self, input_image=None,image_path: str= "./test.png", confidence_threshold: float = 0.5):
        """
        Perform on step object detection on an image using Detectron2.
        """
        p = InferenceBase(self.onstep_detection_cfg,thresh_hold=confidence_threshold)
        print(self.onstep_detection_cfg)

        if input_image is None and image_path is not None:
            input_image = p.read_image(image_path)
        else:
            input_image = pil_image_handler(input_image)
        outputs,vis_output = p.run_on_image(input_image)
        
        return outputs,vis_output
    
    def detect(self,input_image=None, image_path: str = "./test.png", confidence_threshold: float = 0.5):
        """
        Perform object detection on an image using Detectron2.
        """
        p = InferenceBase(self.detection_cfg,thresh_hold=confidence_threshold)
        print(self.detection_cfg)
        if input_image is None and image_path is not None:
            input_image = p.read_image(image_path)
        else:
            input_image = pil_image_handler(input_image)
        outputs,vis_output = p.run_on_image(input_image)

        return outputs,vis_output
    
    def instance_segment(self,input_image=None, image_path: str="./test.png"):
        """
        Perform instance segmentation on an image using Detectron2.
        """
        p = InferenceBase(self.instance_segment_cfg)
        if input_image is None and image_path is not None:
            input_image = p.read_image(image_path)
        else:
            input_image = pil_image_handler(input_image)
        outputs,vis_output = p.run_on_image(input_image)

        return outputs,vis_output
        
    def semantic_segment(self,input_image=None, image_path: str="./test.png"):
        """
        Perform instance segmentation on an image using Detectron2.
        """
    
        p = InferenceBase(self.semantic_segment_cfg,thresh_hold=0.5)
        print(self.semantic_segment_cfg.DATASETS.TEST[0])
        # img = p.read_image(image_path)
        if input_image is None and image_path is not None:
            input_image = Image.open(image_path).convert('RGB')
            input_image = pil_image_handler(input_image)
        outputs,vis_output = p.run_on_image(input_image)

        return outputs,vis_output
    
    def panoptic_segment(self,input_image=None, image_path: str="./test.png"):
        """
        Perform panoptic segmentation on an image using Detectron2.
        """
        p = InferenceBase(self.panoptic_segment_cfg)
        if input_image is None and image_path is not None:
            input_image = p.read_image(image_path)
        else:
            input_image = pil_image_handler(input_image)
        outputs,vis_output = p.run_on_image(input_image)

        # outputs['sem_seg'] = outputs['sem_seg'].numpy().tolist()
        return outputs,vis_output
        
    def keypoint(self, input_image=None,image_path: str="./test.png"):
        """
        Perform keypoint on an image using Detectron2.
        """
        p = InferenceBase(self.keypoints_cfg)
        print(self.detection_cfg)

        if input_image is None and image_path is not None:
            input_image = p.read_image(image_path)
        else:
            input_image = pil_image_handler(input_image)

        outputs,vis_output = p.run_on_image(input_image)

        return outputs,vis_output
        
# if __name__ == "__main__":
#     f = ModelFactory()
#     # f.prepare_meta()
#     out = f.detect()
#     print(out)

    