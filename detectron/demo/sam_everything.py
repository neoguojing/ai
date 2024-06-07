from segment_anything import SamPredictor, sam_model_registry,SamAutomaticMaskGenerator
from PIL import Image
import sys
sys.path.append("..")
from detectron2.data.detection_utils import read_image
from detectron2.utils.visualizer import Visualizer
import numpy as np
from skimage import measure

def seg_with_promp(imput_image,input_prompts):
    sam = sam_model_registry["vit_b"](checkpoint="./sam_vit_b_01ec64.pth")
    predictor = SamPredictor(sam)
    predictor.set_image(imput_image)
    masks, _, _ = predictor.predict(input_prompts)
    return masks

def seg_all(imput_image):
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
        print("segmentation:",obj["segmentation"].shape)
        view.draw_binary_mask(obj["segmentation"])
        
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

if __name__ == "__main__":
    np_image = read_image("./test/cat.jpg")
    print("np_image:",np_image.shape)
    seg_all(np_image)
    