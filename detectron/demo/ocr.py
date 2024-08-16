
from detectron2.data.detection_utils import read_image,pil_image_to_numpy
from detectron2.utils.visualizer import Visualizer
from sam_everything import visimage_to_pil
import numpy as np
def do_ocr(ocr_type,input):
    print(ocr_type)
    result = None
    np_image = pil_image_to_numpy(input)
    if ocr_type == "OCR":
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(lang='ch', use_angle_cls=True)
        # img_path = 'exp.jpeg'
        result = ocr.ocr(np_image)
        print(result)
        result = parse_paddle_result(result)
 
    elif ocr_type == "Easy":
        import easyocr
        reader = easyocr.Reader(['en','ch_sim'])  # 初始化 EasyOCR，选择需要支持的语言（例如英文）
        result = reader.readtext(np_image)
        result = parse_esay_result(result)

    view = Visualizer(np_image)
    for item in result:
        polygon = np.array(item['box'])
        view.draw_polygon(polygon, "k")

    vis_image = view.get_output()
    pil_images = visimage_to_pil([vis_image])
    return pil_images[0],result

def parse_esay_result(data):
    results = []
    for entry in data:
        box = entry[0]
        text = entry[1]
        confidence = entry[2]
        result = {
            'box': box,
            'text': text,
            'confidence': confidence
        }
        results.append(result)
    return results

def parse_paddle_result(data):
    results = []
    for entry in data[0]:
        box = entry[0]
        text = entry[1][0]
        confidence = entry[1][1]
        result = {
            'box': box,
            'text': text,
            'confidence': confidence
        }
        results.append(result)
    return results


