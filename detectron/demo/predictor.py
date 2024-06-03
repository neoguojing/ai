# Copyright (c) Facebook, Inc. and its affiliates.
import atexit
import bisect
import multiprocessing as mp
from collections import deque
import cv2
import torch
import sys
import numpy as np
sys.path.append("..")

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo
from pytorch_predictor import PytorchPredictor
from yolo_predictor import YOLOPredictor
from detectron2.data.detection_utils import convert_PIL_to_numpy
from PIL import Image

class InferenceBase:
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False,device="cpu",thresh_hold=0.5):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        print(self.metadata)
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode
        self.cfg = cfg
        self.cfg.MODEL.DEVICE = device

        self.parallel = parallel

        if cfg.MODEL.WEIGHTS is not None:
            # 用于detection2 内置模型
            if parallel:
                num_gpu = torch.cuda.device_count()
                self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
            else:
                self.predictor = DefaultPredictor(cfg)
        elif cfg.TASK_TYPE is not None:
            # 用于pytorch模型
            if cfg.TASK_TYPE == "yolo":
                self.predictor = YOLOPredictor(cfg)
            else:
                self.predictor = PytorchPredictor(cfg)

        self.output_dir = "./"
        self.thresh_hold = thresh_hold

    def read_image(self,image_path):
        """
        Args:
           image_path: 
        Returns:
            image (np.ndarray):
        """
        from detectron2.data.detection_utils import read_image
        return read_image(image_path)

    def filter_outputs(self, outputs):
        if "instances" not in outputs:
            return outputs
        
        instances = outputs["instances"]
        # 获取每个实例的分数
        scores = instances.scores

        # 创建一个掩码，表示每个实例的分数是否大于阈值
        keep = scores >= self.thresh_hold

        # 根据掩码过滤实例
        filtered_instances = instances[keep]

        # 更新输出的实例
        outputs["instances"] = filtered_instances
        return outputs

    def classes_to_labels(self,classes):
        """
        Returns:
            list[int]:
        """
        class_names = self.metadata.get("thing_classes", None)
        labels = None
        if classes is not None:
            if class_names is not None and len(class_names) > 0:
                labels = [class_names[i] for i in classes]
        return labels

    def plot(self,image,predictions):
        vis_outputs = []
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        if not isinstance(image,np.ndarray):
            image = convert_PIL_to_numpy(image,format=None)

        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
            vis_outputs.append(vis_output)
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
                vis_outputs.append(vis_output)
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)
                vis_outputs.append(vis_output)
            if "sem_segs" in predictions:
                for prediction in predictions["sem_segs"]:
                    vis_output = visualizer.draw_sem_seg(
                        prediction
                    )
                vis_outputs.append(vis_output)
        pil_images = self.visimage_to_pil(vis_outputs)
        return pil_images
    
    def visimage_to_pil(self,visimages):
        pil_images = []
        for visimage in visimages:
            visualized_image = visimage.get_image()[:, :, ::-1]
            pil_image = Image.fromarray(visualized_image)
            pil_images.append(pil_image)
        return pil_images
    
    def save_vis_image(self,visimages):
        import uuid
        for visimage in visimages:
            unique_id = uuid.uuid1()
            visualized_image = visimage.get_image()[:, :, ::-1]
            cv2.imwrite(self.output_dir+str(unique_id)+".png", visualized_image)

    def run_on_image(self,image):
        """
        Args:
            image (np.ndarray or pil image): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_outputs ([VisImage]): the visualized image output.
        """
        if hasattr(self.cfg,"TASK_TYPE") and self.cfg.TASK_TYPE == "yolo":
            predictions,plot_images = self.predictor(image)
            return predictions, plot_images
        else:
            predictions = self.predictor(image)
            predictions = self.filter_outputs(predictions)
            plot_images = self.plot(image,predictions)
            return predictions, plot_images

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def run_on_video(self, video):
        """
        Visualizes predictions on frames of the input video.

        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.

        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, predictions):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                vis_frame = video_visualizer.draw_panoptic_seg_predictions(
                    frame, panoptic_seg.to(self.cpu_device), segments_info
                )
            elif "instances" in predictions:
                predictions = predictions["instances"].to(self.cpu_device)
                vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)
            elif "sem_seg" in predictions:
                vis_frame = video_visualizer.draw_sem_seg(
                    frame, predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )

            # Converts Matplotlib RGB format to OpenCV BGR format
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            return vis_frame

        frame_gen = self._frame_from_video(video)
        if self.parallel:
            buffer_size = self.predictor.default_buffer_size

            frame_data = deque()

            for cnt, frame in enumerate(frame_gen):
                frame_data.append(frame)
                self.predictor.put(frame)

                if cnt >= buffer_size:
                    frame = frame_data.popleft()
                    predictions = self.predictor.get()
                    yield process_predictions(frame, predictions)

            while len(frame_data):
                frame = frame_data.popleft()
                predictions = self.predictor.get()
                yield process_predictions(frame, predictions)
        else:
            for frame in frame_gen:
                yield process_predictions(frame, self.predictor(frame))


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput a little bit when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5

# if __name__ == "__main__":
    
    
#     cfg = get_cfg()
#     cfg.merge_from_file("../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
#     cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
#     cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
#     p = InferenceBase(cfg)
#     img = p.read_image("./test.png")
#     output,image = p.run_on_image(img)
#     print(output)
