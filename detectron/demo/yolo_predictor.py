
from pytorch_model_factory import TorchModelFactory
from PIL import Image
from typing import Any, Dict
import sys
sys.path.append("..")
from detectron2.structures import Instances
import torch
import threading
import gc
from ultralytics import YOLO,solutions
from ultralytics.utils.plotting import Annotator, colors
from detectron2.config import get_cfg

class YOLOPredictor:
    _instances = {}
    _lock = threading.Lock()

    def __new__(cls, cfg=None):
        if cfg is None:
            raise ValueError("Configuration must be provided")
        
        with cls._lock:
            if cfg.TASK_TYPE not in cls._instances:
                cls._instances[cfg.TASK_TYPE] = super(YOLOPredictor, cls).__new__(cls)
                cls._instances[cfg.TASK_TYPE]._initialize(cfg)
        return cls._instances[cfg.TASK_TYPE]

    def _initialize(self, cfg):
        self.cfg = cfg
        if cfg.TASK_TYPE == "classification":
            print("classification")
            self.model = YOLO("yolov8n-cls.pt")
        elif cfg.TASK_TYPE == "detect":
            print("detect")
            self.model = YOLO("yolov8n.pt")
        elif cfg.TASK_TYPE == "pose":
            print("pose")
            self.model = YOLO("yolov8n-pose.pt")
        elif cfg.TASK_TYPE == "obb":
            print("obb")
            self.model = YOLO("yolov8n-obb.pt")
        elif cfg.TASK_TYPE == "instance":
            print("instance")
            self.model = YOLO("yolov8n-seg.pt")
        else:
            print("detect")
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

        if not isinstance(image,list):
            image = [image]

        predictions = self.model(image)
        return self._post_processor(predictions)
    
    def _video_processor(self,video_path,callback=None):
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_path = get_file_path_without_extension(video_path)+"after_inference.mp4"
        print("track:",output_path)
        video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        # Loop through the video frames
        frame_count = 0
        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()
            frame_count+=1
            if success:

                annotated_frame = callback(frame)
                # Display the annotated frame
                # cv2.imshow("YOLOv8 Tracking", annotated_frame)
                video.write(annotated_frame)
                if frame_count % fps == 0:
                    yield None,annotated_frame, None
                
                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # Break the loop if the end of the video is reached
                break

        # Release the video capture object and close the display window
        video.release()
        cap.release()
        # cv2.destroyAllWindows()
        yield None,annotated_frame, output_path
    
    def track(self,video_path):
        def do_track(frame):
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            tracks = self.model.track(frame, persist=True)

            # Visualize the results on the frame
            return tracks[0].plot()
        
        yield from self._video_processor(video_path,do_track)

    def track_with_seg(self,video_path):
        def do_track(frame):
            annotator = Annotator(frame, line_width=2)
            results = self.model.track(frame, persist=True)
            if results[0].boxes.id is not None and results[0].masks is not None:
                masks = results[0].masks.xy
                track_ids = results[0].boxes.id.int().cpu().tolist()

                for mask, track_id in zip(masks, track_ids):
                    annotator.seg_bbox(mask=mask, mask_color=colors(track_id, True), track_label=str(track_id))
            return frame

        yield from self._video_processor(video_path,do_track)

    def counting(self,video_path,region_points=None):
        # Init Object Counter
        counter = solutions.ObjectCounter(
            view_img=False,
            reg_pts=region_points,
            classes_names=self.model.names,
            draw_tracks=True,
            line_thickness=2,
        )

        def do_count(frame):
            tracks = self.model.track(frame, persist=True, show=False)
            return counter.start_counting(frame, tracks)
        
        yield from self._video_processor(video_path,do_count)

    def crop():
        pass

    def gym_monitor(self,video_path,pose_type="pushup"):
        gym_object = solutions.AIGym(
            line_thickness=2,
            view_img=False,
            pose_type=pose_type,
            kpts_to_check=[6, 8, 10],
        )
        def do_gym(frame):
            tracks = self.model.track(frame, persist=True, show=False,verbose=False)
            return gym_object.start_counting(frame, tracks,frame_count=20)
        
        yield from self._video_processor(video_path,do_gym)

    def heatmap():
        pass

    def vision_eye():
        pass

    def speed():
        pass

    def distance():
        pass

    def queue_manager():
        pass
    
    def _post_processor(self, output):
        print("-------yolo------------\n", output)
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
                result["instances"].pred_boxes = o.boxes.xywh

            if o.masks is not None:
                result["instances"].pred_masks = o.masks.xyn

            if o.probs is not None:
                result["instances"].scores = o.probs.top5

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

def get_file_path_without_extension(file_path):
    import os
    # 获取文件所在的目录路径
    directory = os.path.dirname(file_path)
    
    # 获取文件名（包括后缀）
    filename_with_extension = os.path.basename(file_path)
    
    # 分离文件名和后缀
    filename, extension = os.path.splitext(filename_with_extension)
    
    # 拼接目录路径和文件名（不包括后缀）
    return os.path.join(directory, filename)

if __name__ == "__main__":
    cfg = get_cfg()
    cfg.TASK_TYPE = "detect"
    f = YOLOPredictor(cfg)
    # from PIL import Image
    # img = Image.open("./test/test.png")
    f.track("/home/neo/Videos/trafic.webm")

    