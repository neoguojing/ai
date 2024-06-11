from segment_anything import SamPredictor, sam_model_registry,SamAutomaticMaskGenerator
from PIL import Image
import torch
import sys
sys.path.append("..")
from detectron2.data.detection_utils import read_image,pil_image_to_numpy
from detectron2.utils.visualizer import Visualizer
import numpy as np
from skimage import measure

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


class Segmenter:
    def __init__(self, model_name="vit_b", checkpoint_path="./sam_vit_b_01ec64.pth"):
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.sam = sam_model_registry[model_name](checkpoint=checkpoint_path)
        self.predictor = SamPredictor(self.sam)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)

    def seg_with_promp(self, imput_image, point_coords=None, box=None):
        if isinstance(imput_image, Image.Image):
            imput_image = pil_image_to_numpy(imput_image)

        if point_coords is not None:
            point_labels = np.ones(point_coords.shape[0])

        self.predictor.set_image(imput_image)
        masks, _, _ = self.predictor.predict(point_coords=point_coords, point_labels=point_labels, box=box)
        print("seg_with_promp:", masks.shape)
        pil_images = self.draw_bitmask(imput_image, masks)
        return masks, pil_images

    def seg_all(self, imput_image):
        if isinstance(imput_image, Image.Image):
            imput_image = pil_image_to_numpy(imput_image)

        masks = self.mask_generator.generate(imput_image)
        pil_images = self.draw_bitmask(imput_image, masks)
        return masks, pil_images

    def draw_bitmask(self, np_image, masks):
        view = Visualizer(np_image)
        for obj in masks:
            if "segmentation" in obj:
                print("segmentation:", obj["segmentation"].shape)
                view.draw_binary_mask(obj["segmentation"])
            else:
                view.draw_binary_mask(masks[2])

        vis_image = view.get_output()
        pil_images = visimage_to_pil([vis_image])
        return pil_images

    def draw_bitmask_split(self, np_image, masks):
        for i, obj in enumerate(masks):
            print("segmentation:", obj["segmentation"].shape)
            view = Visualizer(np_image)
            view.draw_binary_mask(obj["segmentation"])
            vis_image = view.get_output()
            pil_images = visimage_to_pil([vis_image], idx=i)
        return pil_images

    def draw_polygon(self, np_image, masks):
        view = Visualizer(np_image)
        for obj in masks:
            polygon = bitmask_to_polygon(obj["segmentation"])
            view.draw_polygon(polygon, "k")
        vis_image = view.get_output()
        pil_images = visimage_to_pil([vis_image])
        return pil_images

    @staticmethod
    def bitmask_to_polygon(mask):
        col_mask = np.asfortranarray(mask)
        contours = measure.find_contours(col_mask, 0.5)
        for i, contour in enumerate(contours):
            contour = np.flip(contour, axis=1)
            # polygon = contour.ravel().tolist()
            # print(f"polygon_{i}", polygon)
        return contour


# if __name__ == "__main__":
#     np_image = read_image("./test/face1.jpeg")
#     print("np_image:",np_image.shape)
#     seg_all(np_image)