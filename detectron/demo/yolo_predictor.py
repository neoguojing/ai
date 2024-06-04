
from pytorch_model_factory import TorchModelFactory
# from detectron2.data import MetadataCatalog
import torch
import torchvision.transforms as transforms
from PIL import Image
from typing import Any, Dict
from detectron2.structures import Instances
class YOLOPredictor:

    def __init__(self, cfg=None):
        # self.cfg = cfg.clone()  # cfg can be modified by model
        # self.task_type = cfg.TASK_TYPE
        self.model = TorchModelFactory.create_yolo_detect_model()

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
    
    def _post_processor(self,output):
        print("-------------------\n",output)
        pil_images = []

        result: Dict[str, Instances] = {
            "instances": None
        }

        # TODO 只支持一个图片
        for i,o in enumerate(output):
            # o.save(filename=f"results{i}.jpg")
            im_bgr = o.plot()
            im_rgb = Image.fromarray(im_bgr[..., ::-1])
            pil_images.append(im_rgb)
            
            result["instances"] = Instances(o.orig_shape)

            if o.boxes is not None:
                print(o.boxes.xywh,o.boxes.xywh.shape)
                result["instances"].pred_boxes = o.boxes.xywh

            if o.masks is not None:
                result["instances"].pred_masks = o.masks.xyn

            if o.probs is not None:
                result["instances"].scores = o.probs.top1

            if o.keypoints is not None:
                result["instances"].pred_keypoints = o.keypoints.xyn

            if o.obb is not None:
                result["instances"].pred_obb = o.obb.xywhr

        return result,pil_images


    def release(self):
        import gc
        # 删除模型对象
        del self.model 
        # 清除GPU缓存
        if self.cfg.MODEL.DEVICE == "gpu":
            torch.cuda.empty_cache()
        # 手动触发垃圾回收
        gc.collect()

if __name__ == "__main__":
    f = YOLOPredictor()
    from PIL import Image
    img = Image.open("./test/test.png")
    f(img)

    