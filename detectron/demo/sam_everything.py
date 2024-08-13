from segment_anything import SamPredictor, sam_model_registry,SamAutomaticMaskGenerator
from PIL import Image
import torch
import sys
sys.path.append("..")
from detectron2.data.detection_utils import read_image,pil_image_to_numpy
from detectron2.utils.visualizer import Visualizer
import numpy as np
from skimage import measure
import threading

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def seg_with_promp(imput_image,point_coords=None,box=None):
    if isinstance(imput_image, Image.Image):
        imput_image = pil_image_to_numpy(imput_image)
    point_labels = None
    if point_coords is not None:
        point_labels = np.ones(point_coords.shape[0])
    sam = sam_model_registry["vit_b"](checkpoint="./sam_vit_b_01ec64.pth").to(device)
    predictor = SamPredictor(sam)
    predictor.set_image(imput_image)

    masks = None

    if box is not None:
        masks, _, _ = predictor.predict(box=box)
    elif point_coords is not None and point_labels is not None:
        masks, _, _ = predictor.predict(point_coords=point_coords,point_labels=point_labels)
    print("seg_with_promp:",masks.shape)
    pil_images = draw_bitmask(imput_image,masks)
    return masks,pil_images

def seg_all(imput_image):
    if isinstance(imput_image, Image.Image):
        imput_image = pil_image_to_numpy(imput_image)

    sam = sam_model_registry["vit_b"](checkpoint="./sam_vit_b_01ec64.pth")
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(imput_image)
    pil_images = draw_bitmask(imput_image,masks)
    # pil_images = draw_polygon(imput_image,masks)
    # pil_images = draw_bitmask_split(imput_image,masks)
    return masks,pil_images

# 为每个二值掩码生成一张图片
def draw_bitmask_split(np_image,masks):
    for i,obj in enumerate(masks):
        print("segmentation:",obj["segmentation"].shape)
        view = Visualizer(np_image)
        view.draw_binary_mask(obj["segmentation"])
        vis_image = view.get_output()
        pil_images = visimage_to_pil([vis_image],idx=i)
    return pil_images

# 绘制二值掩码
def draw_bitmask(np_image,masks):
    view = Visualizer(np_image)
    for obj in masks:
        if "segmentation" in obj:
            print("segmentation:",obj["segmentation"].shape)
            view.draw_binary_mask(obj["segmentation"])
        else:
            view.draw_binary_mask(obj)
        
    vis_image = view.get_output()
    pil_images = visimage_to_pil([vis_image])
    return pil_images

# 绘制多边形掩码
def draw_polygon(np_image,masks):
    view = Visualizer(np_image)
    for obj in masks:
        polygon = bitmask_to_polygon(obj["segmentation"])
        view.draw_polygon(polygon,"k")
    vis_image = view.get_output()
    pil_images = visimage_to_pil([vis_image])
    return pil_images


# 二值掩码转换为多边形掩码
def bitmask_to_polygon(mask):
    col_mask = np.asfortranarray(mask)
    contours = measure.find_contours(col_mask,0.5)
    print("contours------",contours.shape)
    for i,contour in enumerate(contours):
        contour = np.flip(contour, axis=1)
        print(f"polygon_{i}",contour.shape)
        # polygon = contour.ravel().tolist()
        # print(f"polygon_{i}",polygon)
    return contour

# VIS图片转换为pil
def visimage_to_pil(visimages,need_save=True,idx=0):
    pil_images = []
    for i,visimage in enumerate(visimages):
        visualized_image = visimage.get_image()
        # [:, :, ::-1]
        pil_image = Image.fromarray(visualized_image)
        if need_save:
            pil_image.save(f"{idx}_{i}.jpg")
        pil_images.append(pil_image)
    return pil_images

def image_to_mask(image, threshold=128):
    # 将图像转换为灰度图像
    if image.mode != 'L':
        image = image.convert('L')
    
    # 将像素值映射到二进制值
    mask_array = np.array(image) > threshold
    
    # 创建一个与原始图像大小相同的数组，用映射后的二进制值填充
    mask_image = Image.fromarray(np.uint8(mask_array) * 255)
    
    return mask_image

class SamAnything:
    _instance = None
    _lock = threading.Lock()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(SamAnything, cls).__new__(cls)
                    cls._instance._initialize(*args, **kwargs)
        return cls._instance

    def _initialize(self, checkpoint_path="./sam_vit_b_01ec64.pth"):
        self.sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path).to(self.device)
        self.predictor = SamPredictor(self.sam)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)

    def seg_with_promp(self, imput_image, point_coords=None, box=None):
        if isinstance(imput_image, Image.Image):
            imput_image = pil_image_to_numpy(imput_image)
        point_labels = None
        if point_coords is not None:
            point_labels = np.ones(point_coords.shape[0])

        self.predictor.set_image(imput_image)
        masks = None

        if box is not None:
            masks, _, _ = self.predictor.predict(box=box)
        elif point_coords is not None and point_labels is not None:
            masks, _, _ = self.predictor.predict(point_coords=point_coords, point_labels=point_labels)

        print("seg_with_promp:", masks.shape)
        pil_images = self.draw_bitmask(imput_image, masks)
        return masks, pil_images

    def seg_all(self, imput_image):
        if isinstance(imput_image, Image.Image):
            imput_image = pil_image_to_numpy(imput_image)

        masks = self.mask_generator.generate(imput_image)
        pil_images = self.draw_bitmask(imput_image, masks)
        return masks, pil_images

    @staticmethod
    def draw_bitmask_split(np_image, masks):
        pil_images = []
        for i, obj in enumerate(masks):
            print("segmentation:", obj["segmentation"].shape)
            view = Visualizer(np_image)
            view.draw_binary_mask(obj["segmentation"])
            vis_image = view.get_output()
            pil_images.extend(SamAnything.visimage_to_pil([vis_image], idx=i))
        return pil_images

    @staticmethod
    def draw_bitmask(np_image, masks):
        view = Visualizer(np_image)
        for obj in masks:
            if "segmentation" in obj:
                print("segmentation:", obj["segmentation"].shape)
                view.draw_binary_mask(obj["segmentation"])
            else:
                view.draw_binary_mask(obj)
        
        vis_image = view.get_output()
        pil_images = SamAnything.visimage_to_pil([vis_image])
        return pil_images

    @staticmethod
    def draw_polygon(np_image, masks):
        view = Visualizer(np_image)
        for obj in masks:
            polygon = SamAnything.bitmask_to_polygon(obj["segmentation"])
            view.draw_polygon(polygon, "k")
        vis_image = view.get_output()
        pil_images = SamAnything.visimage_to_pil([vis_image])
        return pil_images

    @staticmethod
    def bitmask_to_polygon(mask):
        col_mask = np.asfortranarray(mask)
        contours = measure.find_contours(col_mask, 0.5)
        print("contours------", len(contours))
        for i, contour in enumerate(contours):
            contour = np.flip(contour, axis=1)
            print(f"polygon_{i}", contour.shape)
        return contours

    @staticmethod
    def visimage_to_pil(visimages, need_save=True, idx=0):
        pil_images = []
        for i, visimage in enumerate(visimages):
            visualized_image = visimage.get_image()
            pil_image = Image.fromarray(visualized_image)
            if need_save:
                pil_image.save(f"{idx}_{i}.jpg")
            pil_images.append(pil_image)
        return pil_images

    @staticmethod
    def image_to_mask(image, threshold=128):
        if image.mode != 'L':
            image = image.convert('L')
        mask_array = np.array(image) > threshold
        mask_image = Image.fromarray(np.uint8(mask_array) * 255)
        return mask_image


class SamAnything2:
    _instance = None
    _lock = threading.Lock()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(SamAnything, cls).__new__(cls)
                    cls._instance._initialize(*args, **kwargs)
        return cls._instance

    def _initialize(self, checkpoint_path="./sam_vit_b_01ec64.pth"):
        import torch
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-small")
        self.video_predictor = SAM2VideoPredictor.from_pretrained("acebook/sam2-hiera-small")

    def seg_with_promp(self, input_image, video_dir=None,point_coords=None, box=None):
        point_labels = None
        if point_coords is not None:
            point_labels = np.ones(point_coords.shape[0])

        if input_image is not None:
            if isinstance(input_image, Image.Image):
                input_image = pil_image_to_numpy(input_image)
                with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
                    self.predictor.set_image(input_image)
                    masks, _, _ = self.predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        box=box,
                        multimask_output=False)
                    pil_images = self.draw_bitmask(input_image, masks)
                return masks, pil_images
        
        if video_dir is not None:
            with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
                state = self.video_predictor.init_state(video_path=video_dir)
                
                ann_frame_idx = 0  # the frame index we interact with
                ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

                # Let's add a positive click at (x, y) = (210, 350) to get started
                points = np.array([[210, 350]], dtype=np.float32)
                # for labels, `1` means positive click and `0` means negative click
                labels = np.array([1], np.int32)

                # add new prompts and instantly get the output on the same frame
                frame_idx, object_ids, masks = self.video_predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=ann_frame_idx,
                    obj_id=ann_obj_id,
                    points=points,
                    labels=labels,
                    box=box
                )
                video_segments = {} 
                # propagate the prompts to get masklets throughout the video
                for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(state):
                    print(frame_idx,object_ids)
                    video_segments[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }

        print("seg_with_promp:", masks.shape)
        return masks, pil_images

    def seg_all(self, input_image):
        if input_image is not None:
            if isinstance(input_image, Image.Image):
                input_image = pil_image_to_numpy(input_image)

            with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
                self.predictor.set_image(input_image)
                masks, _, _ = self.predictor.predict()
                pil_images = self.draw_bitmask(input_image, masks)
            return masks, pil_images
        

    @staticmethod
    def draw_bitmask_split(np_image, masks):
        pil_images = []
        for i, obj in enumerate(masks):
            print("segmentation:", obj["segmentation"].shape)
            view = Visualizer(np_image)
            view.draw_binary_mask(obj["segmentation"])
            vis_image = view.get_output()
            pil_images.extend(SamAnything.visimage_to_pil([vis_image], idx=i))
        return pil_images

    @staticmethod
    def draw_bitmask(np_image, masks):
        view = Visualizer(np_image)
        for obj in masks:
            if "segmentation" in obj:
                print("segmentation:", obj["segmentation"].shape)
                view.draw_binary_mask(obj["segmentation"])
            else:
                view.draw_binary_mask(obj)
        
        vis_image = view.get_output()
        pil_images = SamAnything.visimage_to_pil([vis_image])
        return pil_images

    @staticmethod
    def draw_polygon(np_image, masks):
        view = Visualizer(np_image)
        for obj in masks:
            polygon = SamAnything.bitmask_to_polygon(obj["segmentation"])
            view.draw_polygon(polygon, "k")
        vis_image = view.get_output()
        pil_images = SamAnything.visimage_to_pil([vis_image])
        return pil_images

    @staticmethod
    def bitmask_to_polygon(mask):
        col_mask = np.asfortranarray(mask)
        contours = measure.find_contours(col_mask, 0.5)
        print("contours------", len(contours))
        for i, contour in enumerate(contours):
            contour = np.flip(contour, axis=1)
            print(f"polygon_{i}", contour.shape)
        return contours

    @staticmethod
    def visimage_to_pil(visimages, need_save=True, idx=0):
        pil_images = []
        for i, visimage in enumerate(visimages):
            visualized_image = visimage.get_image()
            pil_image = Image.fromarray(visualized_image)
            if need_save:
                pil_image.save(f"{idx}_{i}.jpg")
            pil_images.append(pil_image)
        return pil_images

    @staticmethod
    def image_to_mask(image, threshold=128):
        if image.mode != 'L':
            image = image.convert('L')
        mask_array = np.array(image) > threshold
        mask_image = Image.fromarray(np.uint8(mask_array) * 255)
        return mask_image
    
# if __name__ == "__main__":
#     np_image = read_image("./test/face1.jpeg")
#     print("np_image:",np_image.shape)
#     SamAnything.initialize_sam("./sam_vit_b_01ec64.pth")
#     SamAnything.seg_all(np_image)