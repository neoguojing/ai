from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
import os
import json
from PIL import Image

def load_detection2_dataset(data_dir):
    """
    加载自定义的 detection2 数据集
    Args:
        data_dir (str): 数据集根目录
    Returns:
        list[dict]: 包含数据集中每个样本信息的列表
    """
    dataset_dicts = []
    # 假设数据集的注释文件是 JSON 格式
    annotation_file = os.path.join(data_dir, "annotations.json")

    with open(annotation_file) as f:
        data = json.load(f)

    for img_data in data['images']:
        record = {}
        record["file_name"] = os.path.join(data_dir, img_data["file_name"])
        record["height"] = img_data["height"]
        record["width"] = img_data["width"]
        record["image_id"] = img_data["id"]

        objs = []
        for anno in data['annotations']:
            if anno['image_id'] == img_data['id']:
                obj = {
                    "bbox": anno["bbox"],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "category_id": anno["category_id"]
                }
                objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts

def register_my_dataset(data_set_name,data_path,thing_classes):
    for d in ["train", "val"]:
        DatasetCatalog.register(data_set_name + d, lambda d=d: load_detection2_dataset(data_path + d))
        MetadataCatalog.get(data_set_name + d).set(thing_classes=thing_classes)

