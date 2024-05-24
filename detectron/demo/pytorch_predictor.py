
from model_factory import TorchModelFactory
from detectron2.data import MetadataCatalog
import torch
import torchvision.transforms as transforms

class PytorchPredictor:
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a single input image.

    Compared to using the model directly, this class does the following additions:

    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take BGR image as the input and apply conversion defined by `cfg.INPUT.FORMAT`.
    3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
    4. Take one input image and produce a single output, instead of a batch.

    This is meant for simple demo purposes, so it does the above steps automatically.
    This is not meant for benchmarks or running complicated inference logic.
    If you'd like to do anything more complicated, please refer to its source code as
    examples to build and use the model manually.

    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            cfg.DATASETS.TEST.

    Examples:
    ::
        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """

    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.task_type = cfg.TASK_TYPE

        if self.task_type == "class":
            self.model = TorchModelFactory.create_feature_extract_model("resnet")
        elif self.task_type == "feature":
            self.model = TorchModelFactory.create_feature_extract_model("resnet")
        elif self.task_type == "semantic":
            self.modelTorchModelFactory.create_semantic_model("deeplabv3")
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
            if self.metadata.stuff_classes is None:
                self.metadata.stuff_classes = self.metadata.thing_classes
            if len(self.metadata.stuff_classes) == 20:
                self.metadata.stuff_classes.insert(0, "background")
        elif self.task_type == "detect":
            self.model = TorchModelFactory.create_detect_model("Yolo")

    def __call__(self, input_batch):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """

        if self.model is None:
            return None
        
        if torch.cuda.is_available():
            input_batch = input_batch.cuda()
        
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            if self.model is None:
                return None
        
            predictions = self.model(input_batch)
            return self._post_processor(predictions)
        
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
    
    def _post_processor(self,output):
        result = None
        if self.task_type == "class":
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
            return result
        elif self.task_type == "semantic":
            output = output["out"]
            output_predictions = output.argmax(1)
            output_predictions = output_predictions.cpu()
            result = {"sem_seg":output_predictions}
            return result
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