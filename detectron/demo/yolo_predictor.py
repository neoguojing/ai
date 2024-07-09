
from pytorch_model_factory import TorchModelFactory
from PIL import Image
from typing import Any, Dict
from detectron2.structures import Instances
import torch
import threading
import gc
from ultralytics import YOLO

class YOLOPredictor:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, cfg=None):
        if cfg is None:
            raise ValueError("Configuration must be provided")
        
        with cls._lock:
            if cfg not in cls._instances:
                cls._instances[cfg] = super(YOLOPredictor, cls).__new__(cls)
                cls._instances[cfg]._initialize(cfg)
        return cls._instances[cfg]

    def _initialize(self, cfg):
        self.cfg = cfg
        if cfg.TASK_TYPE == "classfication":
            self.model = YOLO("yolov8n-cls.pt")
        elif cfg.TASK_TYPE == "detect":
            self.model = YOLO("yolov8n.pt")
        elif cfg.TASK_TYPE == "pose":
            self.model = YOLO("yolov8n-pose.pt")
        elif cfg.TASK_TYPE == "obb":
            self.model = YOLO("yolov8n-obb.pt")
        elif cfg.TASK_TYPE == "segment":
            self.model = YOLO("yolov8n-seg.pt")
        else:
            self.model = YOLO("yolov8n.pt")

    # def __new__(cls, cfg=None):
    #     if cls._instance is None:
    #         with cls._lock:
    #             if cls._instance is None:
    #                 cls._instance = super(YOLOPredictor, cls).__new__(cls)
    #                 cls._instance._initialize(cfg)
    #     return cls._instance

    # def _initialize(self, cfg=None):
    #     self.model = TorchModelFactory.create_yolo_detect_model()

    def __call__(self, image):
        """
        Args:
            image (PIL image): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """

        if self.model is None:
            return None
    
        predictions = self.model([image])
        return self._post_processor(predictions)
    
    def _post_processor(self, output):
        print("-------------------\n", output)
        pil_images = []

        result: Dict[str, Instances] = {
            "instances": None
        }

        # TODO 只支持一个图片
        for i, o in enumerate(output):
            im_bgr = o.plot()
            im_rgb = Image.fromarray(im_bgr[..., ::-1])
            pil_images.append(im_rgb)
            
            result["instances"] = Instances(o.orig_shape)

            if o.boxes is not None:
                print(o.boxes.xywh, o.boxes.xywh.shape)
                result["instances"].pred_boxes = o.boxes.xywh

            if o.masks is not None:
                result["instances"].pred_masks = o.masks.xyn

            if o.probs is not None:
                result["instances"].scores = o.probs.top1

            if o.keypoints is not None:
                result["instances"].pred_keypoints = o.keypoints.xyn

            if o.obb is not None:
                result["instances"].pred_obb = o.obb.xywhr

        return result, pil_images

    def release(self):
        # 删除模型对象
        del self.model 
        # 清除GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # 手动触发垃圾回收
        gc.collect()



# if __name__ == "__main__":
#     f = YOLOPredictor()
#     from PIL import Image
#     img = Image.open("./test/test.png")
#     f(img)

    