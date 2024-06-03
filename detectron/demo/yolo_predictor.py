
from pytorch_model_factory import TorchModelFactory
# from detectron2.data import MetadataCatalog
import torch
import torchvision.transforms as transforms

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
        for i,o in enumerate(output):
            o.save(filename=f"results{i}.jpg")

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

    