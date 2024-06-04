from segment_anything import SamPredictor, sam_model_registry,SamAutomaticMaskGenerator
import sys
sys.path.append("..")
from detectron2.data.detection_utils import read_image

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
    print(masks)
    return masks


if __name__ == "__main__":
    np_image = read_image("./test/cat.jpg")
    seg_all(np_image)