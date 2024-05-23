
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
        self.semantic_segment_cfg = ModelConfig(ModelCategory.SEMANTIC_SEGMENTATION, 
                                        #  model_path="PascalVOC-Detection/faster_rcnn_R_50_FPN.yaml",
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
        
        model = TorchModelFactory.create_feature_extract_model("resnet")

    
        input_batch = self.image_processor(image_path)

        with torch.no_grad():
            output = model(input_batch)
            
        result = {"features":output}
        # print(result)
        return result
    
    def classify(self, image_path: str="./test.png"):
        """
        Perform classification on an image using Detectron2.
        """
        model = TorchModelFactory.create_feature_extract_model("resnet")

        input_batch = self.image_processor(image_path)

        with torch.no_grad():
            output = model(input_batch)
        
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        # print(probabilities,probabilities.shape)
        top5_prob, top5_catid = torch.topk(probabilities, 1)
        # print(top5_prob,top5_catid)
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
        input_batch = self.image_processor(image_path,resize=None,crop=None)

        with torch.no_grad():
            output = model(input_batch)["out"][0]
        print("output shape",output.shape)
        output_predictions = output.argmax(0)
        print(output_predictions)
        print("output_predictions shape",output_predictions.shape)

        from detectron2.data.detection_utils import read_image
        import uuid
        image = read_image(image_path)
        voc_train_metadata = self.prepare_meta("voc")
        print(voc_train_metadata)
        visualizer = Visualizer(image,metadata=voc_train_metadata)

        visimage = visualizer.draw_sem_seg(output_predictions)                    
        unique_id = uuid.uuid1()
        visualized_image = visimage.get_image()[:, :, ::-1]
        cv2.imwrite("./"+str(unique_id)+".png", visualized_image)
        return output_predictions
    
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
    f.prepare_meta()
    # f.classify()

    