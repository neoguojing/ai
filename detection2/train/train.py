import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import ColorMode

class ModelTrainer:
    def __init__(self, model_config_path, class_names, task="object_detection"):
        """
        Initializes the trainer for various tasks.

        Args:
            model_config_path (str): Path to the model configuration file.
            class_names (list): List of class names for the dataset.
            task (str): The type of task to perform, e.g., "object_detection".
        """
        self.model_config_path = model_config_path
        self.class_names = class_names
        self.task = task
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_config_path)
        
        if task == "object_detection":
            self.cfg.DATASETS.TRAIN = ("balloon_train",)
            self.cfg.DATASETS.TEST = ()
            self.cfg.DATALOADER.NUM_WORKERS = 2
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.model_config_path)
            self.cfg.SOLVER.IMS_PER_BATCH = 2
            self.cfg.SOLVER.BASE_LR = 0.00025
            self.cfg.SOLVER.MAX_ITER = 300
            self.cfg.SOLVER.STEPS = []
            self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
            self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_names)
        else:
            raise ValueError("Unsupported task: {}".format(task))
        
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)

    def train(self):
        """
        Trains the model using the provided configuration.
        """
        if self.task == "object_detection":
            trainer = DefaultTrainer(self.cfg)
            trainer.resume_or_load(resume=False)
            trainer.train()
        else:
            raise ValueError("Unsupported task: {}".format(self.task))

if __name__ == "__main__":
    # Use the class to train a object detection model with 2 classes
    class_names = ["balloon", "person"]
    trainer = ModelTrainer("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", class_names, task="object_detection")
    trainer.train()
