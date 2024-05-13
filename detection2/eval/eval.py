import detectron2
from detectron2.utils.logger import setup_logger

class ModelEvaluator:
    def __init__(self, model_path: str, dataset_name: str):
        setup_logger()

        # Import necessary libraries
        import numpy as np
        import os, json, cv2, random
        from google.colab.patches import cv2_imshow

        # Import detectron2 utilities
        from detectron2 import model_zoo
        from detectron2.engine import DefaultPredictor
        from detectron2.config import get_cfg
        from detectron2.utils.visualizer import Visualizer
        from detectron2.data import MetadataCatalog, DatasetCatalog
        from detectron2.utils.visualizer import ColorMode
        from detectron2.evaluation import COCOEvaluator, inference_on_dataset
        from detectron2.data import build_detection_test_loader

        # Load configuration and model
        self.cfg = get_cfg()
        self.predictor = DefaultPredictor(model_path)
        self.dataset_name = dataset_name

        # Register dataset
        DatasetCatalog.register(dataset_name, lambda: load_detection2_dataset("/path/to/detection2_dataset"))
        MetadataCatalog.get(dataset_name).set(thing_classes=["class1", "class2", "class3"])

    def evaluate(self):
        """
        Evaluates the model on the specified dataset.
        """
        # Evaluate model
        evaluator = COCOEvaluator(self.dataset_name, output_dir="./output")
        val_loader = build_detection_test_loader(self.cfg, self.dataset_name)
        print(inference_on_dataset(self.predictor.model, val_loader, evaluator))

# Example usage
evaluator = ModelEvaluator("path/to/model.pth", "my_dataset")
evaluator.evaluate()
