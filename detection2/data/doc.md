# 常用字段（Common Fields）
- `file_name` (str): 图像文件的完整路径。
- `height`, `width` (int): 图像的高度和宽度，表示图像的形状。
- `image_id` (str 或 int): 表示图像的唯一标识符，用于识别图像。

## 实例检测/分割任务字段（Instance Detection/Segmentation Fields）
- `annotations` (list[dict]): 包含每个图像中实例的注释信息列表。
- `bbox` (list[float], required): 表示实例边界框的四个坐标值，格式为 `[x_min, y_min, x_max, y_max]`。
- `bbox_mode` (int, required): 边界框坐标的格式，支持 `BoxMode.XYXY_ABS` 或 `BoxMode.XYWH_ABS`。
- `category_id` (int, required): 表示实例的类别标签，范围为 `[0, num_categories-1]`，其中 `num_categories` 通常用于表示“背景”类别。
- `segmentation` (list[list[float]] or dict): 表示实例的分割掩码，可以是多边形列表或 COCO 压缩的 RLE 格式。
- `keypoints` (list[float]): 表示实例关键点的坐标，以及可见性信息。

## 语义分割任务字段（Semantic Segmentation Fields）
- `sem_seg_file_name` (str): 语义分割标注文件的完整路径，是一个灰度图像，像素值为整数标签。

## 全景分割任务字段（Panoptic Segmentation Fields）
- `pan_seg_file_name` (str): 全景分割标注文件的完整路径，是一个 RGB 图像，像素值为整数 id，使用 `panopticapi.utils.id2rgb` 函数编码。
- `segments_info` (list[dict]): 定义全景分割标注中每个 id 的含义，每个 dict 包含以下字段：
  - `id` (int): 在标注图像中出现的整数 id。
  - `category_id` (int): 表示 id 对应的类别标签。


# 通用元数据键（Common Metadata Keys）
- `thing_classes` (list[str]): 所有实例检测/分割任务都会用到。每个实例/物体类别的名称列表。如果加载 COCO 格式的数据集，会自动由 `load_coco_json` 函数设置。
- `thing_colors` (list[tuple(r, g, b)]): 每个实例/物体类别的预定义颜色，用于可视化。如果未提供，将使用随机颜色。
- `stuff_classes` (list[str]): 语义分割和全景分割任务使用。每个 stuff 类别的名称列表。
- `stuff_colors` (list[tuple(r, g, b)]): 每个 stuff 类别的预定义颜色，用于可视化。如果未提供，将使用随机颜色。
- `ignore_label` (int): 语义分割和全景分割任务使用。标注中具有此类别标签的像素将在评估中被忽略，通常用于表示“未标记”像素。
- `keypoint_names` (list[str]): 关键点检测任务使用。每个关键点的名称列表。
- `keypoint_flip_map` (list[tuple[str]]): 关键点检测任务使用。一个关键点名称的对列表，表示在图像水平翻转时应该翻转的两个关键点。
- `keypoint_connection_rules` (list[tuple(str, str, (r, g, b))]): 关键点检测任务使用。每个元组指定连接的两个关键点以及在可视化时用于连接线的颜色。

## 添加的特定评估元数据键（Additional Metadata for Evaluation）
以下元数据键通常用于特定数据集的评估（例如 COCO 数据集）：
- `thing_dataset_id_to_contiguous_id` (dict[int->int]): COCO 格式数据集中实例类别 id 到连续 id 的映射，用于评估时将类别 id 对齐到连续的范围。
- `stuff_dataset_id_to_contiguous_id` (dict[int->int]): 用于语义分割和全景分割任务，在生成预测 JSON 文件时将语义分割类别 id 对齐到连续的范围。
- `json_file` (str): COCO 注释的 JSON 文件路径，用于 COCO 格式数据集的评估。
- `panoptic_root`, `panoptic_json` (str): COCO 格式全景分割评估时用到的全景分割根目录和 JSON 文件路径。
- `evaluator_type` (str): 内置主训练脚本使用的评估器类型，不建议在新的训练脚本中使用，可以直接在主脚本中为数据集提供相应的 `DatasetEvaluator`。
