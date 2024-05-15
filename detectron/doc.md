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


# 内置模型的输入格式
内置模型支持的输入格式是一个包含多个字典的列表，每个字典对应于一张图像的信息。每个字典可以包含以下键：

- `"image"`：表示图像的张量，格式为 (C, H, W)。通道的含义由配置 `cfg.INPUT.FORMAT` 定义。在模型内部使用配置 `cfg.MODEL.PIXEL_{MEAN,STD}` 进行图像标准化（如果有的话）。

- `"height"`, `"width"`：推理时期望的输出高度和宽度。这与图像字段的高度或宽度不一定相同。例如，图像字段包含调整大小后的图像（如果使用了调整大小作为预处理步骤）。但是你可能希望输出的结果是在原始分辨率上。如果提供了这些字段，模型将以这个分辨率产生输出，而不是在输入模型的图像的分辨率上。这样做更加高效和准确。

- `"instances"`：用于训练的 Instances 对象，包含以下字段：

  - `"gt_boxes"`：一个 Boxes 对象，存储了 N 个边界框，每个边界框对应一个实例。
  
  - `"gt_classes"`：长整型张量，长度为 N，存储了 N 个实例的标签，范围在 [0, num_categories) 内。

  - `"gt_masks"`：一个 PolygonMasks 或 BitMasks 对象，存储了 N 个掩码，每个掩码对应一个实例。

  - `"gt_keypoints"`：一个 Keypoints 对象，存储了 N 个关键点集合，每个集合对应一个实例。

  - `"sem_seg"`：整型张量，格式为 (H, W)，用于训练的语义分割标签。数值从 0 开始，代表不同的类别标签。

  - `"proposals"`：仅用于 Fast R-CNN 风格模型的 Instances 对象，包含以下字段：

    - `"proposal_boxes"`：一个 Boxes 对象，存储了 P 个提议框。

    - `"objectness_logits"`：一个张量，长度为 P，存储了 P 个提议框的得分。

对于内置模型的推理，只需要提供 `"image"` 键即可，而 `"width/height"` 是可选的。

目前我们没有为全景分割训练定义标准的输入格式，因为模型现在使用由自定义数据加载器生成的自定义格式。

# 内置模型输出格式

在训练模式下，内置模型输出一个包含所有损失的 dict[str->ScalarTensor]。

在推理模式下，内置模型输出一个包含多个字典的列表，每个字典对应每张图像的输出结果。根据模型执行的任务不同，每个字典可能包含以下字段：

- `"instances"`：Instances 对象，包含以下字段：

  - `"pred_boxes"`：Boxes 对象，存储了 N 个预测边界框，每个对应一个检测实例。

  - `"scores"`：张量，长度为 N，存储了 N 个实例的置信度分数。

  - `"pred_classes"`：张量，长度为 N，存储了 N 个预测标签，范围在 [0, num_categories) 内。

  - `"pred_masks"`：形状为 (N, H, W) 的张量，存储了每个检测实例的掩码。

  - `"pred_keypoints"`：形状为 (N, num_keypoint, 3) 的张量，每个关键点的表示为 (x, y, score)。置信度分数大于 0。

- `"sem_seg"`：形状为 (num_categories, H, W) 的张量，语义分割的预测结果。

- `"proposals"`：Instances 对象，包含以下字段：

  - `"proposal_boxes"`：Boxes 对象，存储了 N 个提议框。

  - `"objectness_logits"`：长度为 N 的张量，存储了 N 个提议框的置信度分数。

- `"panoptic_seg"`：一个元组 (pred: Tensor, segments_info: Optional[list[dict]])，其中 `pred` 张量的形状为 (H, W)，包含每个像素的分割 id。

  如果 `segments_info` 存在，则每个字典描述了一个分割 id，并包含以下字段：

  - `"id"`：分割 id
  
  - `"isthing"`：分割是否为物体或背景
  
  - `"category_id"`：该分割的类别 id。

  如果某个像素的 id 在 `segments_info` 中不存在，则视为全景分割中的无效标签。

  如果 `segments_info` 为 None，则 `pred` 中的所有像素值必须 ≥ -1。值为 -1 的像素被视为无效标签。否则，每个像素的类别 id 可以通过 `category_id = pixel // metadata.label_divisor` 获得。

