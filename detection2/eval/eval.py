import detectron2
from detectron2.utils.logger import setup_logger

def evaluate_model(model_path: str, dataset_name: str):
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
    cfg = get_cfg()
    predictor = DefaultPredictor(model_path)

    # Register dataset
    DatasetCatalog.register(dataset_name, lambda: load_detection2_dataset("/path/to/detection2_dataset"))
    MetadataCatalog.get(dataset_name).set(thing_classes=["class1", "class2", "class3"])

    # Evaluate model
    evaluator = COCOEvaluator(dataset_name, output_dir="./output")
    val_loader = build_detection_test_loader(cfg, dataset_name)
    print(inference_on_dataset(predictor.model, val_loader, evaluator))

# Example usage
evaluate_model("path/to/model.pth", "my_dataset")
