
from model_factory import TorchModelFactory
from detectron2.data import MetadataCatalog
import torch
import torchvision.transforms as transforms

class PytorchPredictor:

    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.task_type = cfg.TASK_TYPE
        self.resize = 256
        self.crop = 224
        if self.task_type == "classfication":
            self.model = TorchModelFactory.create_feature_extract_model("resnet")
        elif self.task_type == "feature":
            self.model = TorchModelFactory.create_feature_extract_model("resnet")
        elif self.task_type == "semantic":
            self.model = TorchModelFactory.create_semantic_model("deeplabv3")
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
            if not hasattr(self.metadata,"stuff_classes"):
                self.metadata.stuff_classes = self.metadata.thing_classes
            if len(self.metadata.stuff_classes) == 20:
                self.metadata.stuff_classes.insert(0, "background")
            
            print(self.metadata)
            self.resize = None
            self.crop = None
        elif self.task_type == "detect":
            self.model = TorchModelFactory.create_detect_model("Yolo")

    def __call__(self, image):
        """
        Args:
            image (tensor): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """

        if self.model is None:
            return None
        
        image = self.image_processor(image)
        input_batch = image.unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.cuda()
        
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            if self.model is None:
                return None
        
            predictions = self.model(input_batch)
            return self._post_processor(predictions)
        
    def image_processor(self,input_image):
        # from PIL import Image
        # input_image = Image.open(image_path).convert('RGB')
        preprocess = transforms.Compose([
            transforms.Resize(self.resize) if self.resize is not None else lambda x: x,
            transforms.CenterCrop(self.crop) if self.crop is not None else lambda x: x,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        
        return input_tensor
    
    def _post_processor(self,output):
        result = None
        if self.task_type == "classfication":
            output = output.cpu()
            result = {"classfication":[]}
            probabilities = torch.nn.functional.softmax(output, dim=1)
            for i, probabilitiy in enumerate(probabilities):
                top5_prob, top5_catid = torch.topk(probabilitiy, 1)
                target = {"feature":output[i],"score":top5_prob,"pred_class":top5_catid}
                result["classfication"].append(target)   
        elif self.task_type == "feature":
            output = output.cpu()
            result = {"features":output}
        elif self.task_type == "semantic":
            output = output["out"]
            output_predictions = output.argmax(1)
            output_predictions = output_predictions.cpu()
            result = {"sem_segs":output_predictions}
        elif self.task_type == "detect":
            pass
    
        return result

    def release(self):
        import gc
        # 删除模型对象
        del self.model 
        # 清除GPU缓存
        if self.cfg.MODEL.DEVICE == "gpu":
            torch.cuda.empty_cache()
        # 手动触发垃圾回收
        gc.collect()