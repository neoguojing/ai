from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
import os
import json

class CustomDataset:
    def __init__(self, data_dir, dataset_name='my_dataset', thing_classes=None):
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.thing_classes = thing_classes
        self.data = self._load_annotations()
        self.train_dataset_dicts = []
        self.val_dataset_dicts = []

    def _load_annotations(self):
        annotation_file = os.path.join(self.data_dir, "annotations.json")
        with open(annotation_file) as f:
            data = json.load(f)
        return data

    def _create_image_record(self, img_data, data_dir):
        record = {
            "file_name": os.path.join(data_dir, img_data["file_name"]),
            "height": img_data["height"],
            "width": img_data["width"],
            "image_id": img_data["id"],
            "annotations": []
        }
        return record

    def _add_annotations_to_record(self, record, annotations):
        for anno in annotations:
            if anno['image_id'] == record['image_id']:
                obj = {
                    "bbox": anno["bbox"],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "category_id": anno["category_id"]
                }
                record["annotations"].append(obj)

    def _create_dataset_dicts(self, data_type):
        dataset_dicts = []
        data_dir = os.path.join(self.data_dir, data_type)

        for img_data in self.data['images']:
            record = self._create_image_record(img_data, data_dir)
            self._add_annotations_to_record(record, self.data['annotations'])
            dataset_dicts.append(record)

        return dataset_dicts

    def register_datasets(self):
        for d in ["train", "val"]:
            dataset_dicts = self._create_dataset_dicts(d)
            if d == "train":
                self.train_dataset_dicts = dataset_dicts
            elif d == "val":
                self.val_dataset_dicts = dataset_dicts

            DatasetCatalog.register(self.dataset_name + d, lambda d=d: self._create_dataset_dicts(d))
            MetadataCatalog.get(self.dataset_name + d).set(thing_classes=self.thing_classes)

if __name__ == "__main__":
    # Example usage:
    data_dir = "/path/to/your/data/"
    thing_classes = ["class1", "class2", "class3"]
    dataset_name = 'my_dataset'
    custom_dataset = CustomDataset(data_dir, dataset_name, thing_classes)
    custom_dataset.register_datasets()

    # Access the dataset dicts if needed
    train_dataset = DatasetCatalog.get(dataset_name + "train")
    val_dataset = DatasetCatalog.get(dataset_name + "val")
