# 模型配置

- `_C.MODEL.LOAD_PROPOSALS = False`: 设置是否加载预定义的提案。这决定了在训练或推断期间是否需要加载预生成的提案。
- `_C.MODEL.MASK_ON = False`: 指定是否应该启用掩膜预测。如果为 `True`，则模型将生成对象掩膜。
- `_C.MODEL.KEYPOINT_ON = False`: 指定是否启用关键点检测。如果设置为 `True`，则模型将进行关键点的预测。
- `_C.MODEL.DEVICE = "cuda"`: 指定模型在哪个设备上运行，这里设定为 `CUDA`，即使用 GPU 进行加速。
- `_C.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"`: 指定模型的元架构，这里使用通用的 RCNN 架构。
- `_C.MODEL.WEIGHTS = ""`: 指定加载预训练权重的路径。你可以设置一个文件路径或者 URL，比如模型动物园（model zoo）中的权重。
- `_C.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675]`: 图像像素均值，用于图像归一化。这些值是根据 ImageNet 数据集的平均像素值计算的。
- `_C.MODEL.PIXEL_STD = [1.0, 1.0, 1.0]`: 图像像素标准差，用于图像归一化。这些值被设置为 `1.0`，这是因为在某些预训练模型中，标准差已经被吸收到了权重中，所以这里设置为 `1.0`。

# 输入配置

- `_C.INPUT.MIN_SIZE_TRAIN = (800,)`: 训练期间图像最小边的大小。这里设置为 `(800,)` 表示图像将被调整为最小边至少为 800 像素。
- `_C.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"`: 训练期间最小边大小的采样方式。这里设置为 `choice`，表示可以从 `INPUT.MIN_SIZE_TRAIN` 中选择一个值进行采样。
- `_C.INPUT.MAX_SIZE_TRAIN = 1333`: 训练期间图像最大边的大小。图像将调整为最大边不超过 1333 像素。
- `_C.INPUT.MIN_SIZE_TEST = 800`: 测试期间图像的最小边大小。设置为 800 表示测试时图像将被调整为最小边至少为 800 像素。
- `_C.INPUT.MAX_SIZE_TEST = 1333`: 测试期间图像的最大边大小。图像将调整为最大边不超过 1333 像素。
- `_C.INPUT.RANDOM_FLIP = "horizontal"`: 数据增强过程中图像翻转的方式。这里设置为 `horizontal` 表示训练时可以随机水平翻转图像。
- `_C.INPUT.CROP.ENABLED = False`: 是否启用数据增强中的裁剪操作。默认为 `False`，表示不使用裁剪。
- `_C.INPUT.CROP.TYPE = "relative_range"`: 裁剪操作的类型。这里设置为 `relative_range`，表示裁剪尺寸相对于原始图像尺寸的比例范围。
- `_C.INPUT.CROP.SIZE = [0.9, 0.9]`: 裁剪操作的尺寸设置。如果 `CROP.TYPE` 是 `relative_range`，则表示裁剪尺寸相对于原始图像尺寸的比例范围。
- `_C.INPUT.FORMAT = "BGR"`: 图像输入格式。这里设置为 `BGR`，表示图像使用 BGR 格式，但在内部将会转换为 RGB。
- `_C.INPUT.MASK_FORMAT = "polygon"`: 模型使用的真值掩码格式。这里设置为 `polygon`，表示模型将使用多边形表示掩码。

# 数据配置

- `_C.DATASETS.TRAIN = ()`: 训练时要使用的数据集名称列表，这些数据集必须在 DatasetCatalog 中注册。训练过程将合并并使用这些数据集中的样本作为训练数据。
- `_C.DATASETS.PROPOSAL_FILES_TRAIN = ()`: 训练时用于预先计算的提议文件列表，这些文件必须与 `DATASETS.TRAIN` 中列出的数据集一致。
- `_C.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN = 2000`: 训练过程中保留的预先计算提议的最高分数的数量。
- `_C.DATASETS.TEST = ()`: 测试时要使用的数据集名称列表，这些数据集必须在 DatasetCatalog 中注册。
- `_C.DATASETS.PROPOSAL_FILES_TEST = ()`: 测试时用于预先计算的提议文件列表，这些文件必须与 `DATASETS.TEST` 中列出的数据集一致。
- `_C.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST = 1000`: 测试过程中保留的预先计算提议的最高分数的数量。

# 数据加载配置

- `_C.DATALOADER.NUM_WORKERS = 4`: 数据加载器使用的数据加载线程数。
- `_C.DATALOADER.ASPECT_RATIO_GROUPING = True`: 如果为 `True`，则每个批次应仅包含宽高比兼容的图像。这样可以将纵向图像分组在一起，横向图像不会与纵向图像混合在同一批次中。
- `_C.DATALOADER.SAMPLER_TRAIN = "TrainingSampler"`: 用于训练数据集的采样器的类型，可以是 `TrainingSampler` 或 `RepeatFactorTrainingSampler`。
- `_C.DATALOADER.REPEAT_THRESHOLD = 0.0`: `RepeatFactorTrainingSampler` 的重复阈值。
- `_C.DATALOADER.REPEAT_SQRT = True`: 如果为 `True`，在计算重复因子时会使用平方根。
- `_C.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True`: 如果为 `True`，在处理具有实例注释的数据集时，训练数据加载器将过滤掉没有关联注释的图像。

# 主干配置

- `_C.MODEL.BACKBONE.NAME = "build_resnet_backbone"`: 指定模型主干的名称，这里使用了 `build_resnet_backbone`，表示构建一个 ResNet 主干。
- `_C.MODEL.BACKBONE.FREEZE_AT = 2`: 冻结模型主干的前几个阶段，使其不进行训练。在 ResNet 中，共有 5 个阶段，第一个阶段是一个卷积层，后续阶段是一组组残差块。设置为 `2` 表示冻结前两个阶段，即冻结卷积层和第一个组的残差块，而后面的阶段将参与训练过程。

# FPN 多尺度特征提取

- `_C.MODEL.FPN.IN_FEATURES`: 这是用于 FPN 的输入特征图的名称列表。这些特征图通常是来自底层网络的不同层级的特征，具有连续的对数级别的步长（strides）。例如，`["res2", "res3", "res4", "res5"]` 表示从底层到高层的特征图名称列表。
- `_C.MODEL.FPN.OUT_CHANNELS`: FPN 输出的特征通道数。这指定了 FPN 输出特征图的通道数，通常为 `256`。
- `_C.MODEL.FPN.NORM`: FPN 中特征融合过程中使用的归一化方式。可以选择不使用归一化 (`""`) 或使用 Group Normalization (`"GN"`)。
- `_C.MODEL.FPN.FUSE_TYPE`: FPN 中特征融合的方式。可以选择 `"sum"` 或 `"avg"`，分别表示特征融合使用求和或求平均的方式。

# 提案生成器配置：目标提案或候选框

- `_C.MODEL.PROPOSAL_GENERATOR.NAME`: 指定了使用的提案生成器的名称。目前支持的选项有 "RPN"、"RRPN" 和 "PrecomputedProposals"。这些生成器用于在目标检测模型中生成候选提案。
- `_C.MODEL.PROPOSAL_GENERATOR.MIN_SIZE`: 指定了生成的提案的高度和宽度需要大于的最小尺寸阈值。这个阈值通常是在训练或推理期间使用的尺度。设置为 `0` 表示没有最小尺寸限制。

# 锚点生成：用于在目标检测模型中生成锚点，这些锚点是用于生成候选提案的基础

- `_C.MODEL.ANCHOR_GENERATOR.NAME`: 指定了使用的锚点生成器的名称，可以是注册在 `ANCHOR_GENERATOR` 注册表中的任何名称。
- `_C.MODEL.ANCHOR_GENERATOR.SIZES`: 指定了生成锚点的大小（面积的平方根）列表，以绝对像素为单位相对于网络输入。格式为 `list[list[float]]`，其中 `SIZES[i]` 指定了用于 `IN_FEATURES[i]` 的尺寸列表。`len(SIZES)` 必须等于 `len(IN_FEATURES)` 或为 1。当 `len(SIZES) == 1` 时，`SIZES[0]` 将用于所有 `IN_FEATURES`。
- `_C.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS`: 指定了用于生成锚点的不同长宽比列表。对于给定的 `SIZES` 中的每个面积，通过锚点生成器生成具有不同长宽比的锚点。格式为 `list[list[float]]`，其中 `ASPECT_RATIOS[i]` 指定了用于 `IN_FEATURES[i]` 的长宽比列表。`len(ASPECT_RATIOS)` == `len(IN_FEATURES)` 必须成立，或者 `len(ASPECT_RATIOS) == 1`，并且使用 `ASPECT_RATIOS[0]` 列表作为所有 `IN_FEATURES` 的长宽比。
- `_C.MODEL.ANCHOR_GENERATOR.ANGLES`: 指定了生成锚点时考虑的角度列表（以度为单位）。格式为 `list[list[float]]`，其中 `ANGLES[i]` 指定了用于 `IN_FEATURES[i]` 的角度列表。
- `_C.MODEL.ANCHOR_GENERATOR.OFFSET`: 指定了第一个锚点中心与图像左上角之间的相对偏移量，取值范围为 `[0, 1)`。推荐使用 `0.5`，表示半个步长。这个值预计不会影响模型的准确性。

### RPN (Region Proposal Network) 选项
- `MODEL.RPN.HEAD_NAME`: RPN头部的名称，用于`RPN_HEAD_REGISTRY`。
- `MODEL.RPN.IN_FEATURES`: RPN使用的输入特征图名称列表。
- `MODEL.RPN.BOUNDARY_THRESH`: 移除超出图像边界的RPN锚点的阈值。
- `MODEL.RPN.IOU_THRESHOLDS`: RPN正负样本的IoU重叠阈值。
- `MODEL.RPN.IOU_LABELS`: 每个IoU阈值对应的标签。
- `MODEL.RPN.BATCH_SIZE_PER_IMAGE`: 每个图像用于训练RPN的候选区域数。
- `MODEL.RPN.POSITIVE_FRACTION`: 前景样本的比例。
- `MODEL.RPN.BBOX_REG_LOSS_TYPE`: 边界框回归损失类型。
- `MODEL.RPN.BBOX_REG_LOSS_WEIGHT`: 边界框回归损失权重。
- `MODEL.RPN.BBOX_REG_WEIGHTS`: 归一化RPN锚点回归目标的权重。
- `MODEL.RPN.SMOOTH_L1_BETA`: 平滑L1损失的转折点。
- `MODEL.RPN.PRE_NMS_TOPK_TRAIN/TEST`: NMS前保留的topk候选区域数。
- `MODEL.RPN.POST_NMS_TOPK_TRAIN/TEST`: NMS后保留的topk候选区域数。
- `MODEL.RPN.NMS_THRESH`: NMS阈值。
- `MODEL.RPN.CONV_DIMS`: RPN卷积维度。

### ROI HEADS (Region of Interest Heads) 选项
- `MODEL.ROI_HEADS.NAME`: ROI heads的名称。
- `MODEL.ROI_HEADS.NUM_CLASSES`: 前景类别数。
- `MODEL.ROI_HEADS.IN_FEATURES`: ROI heads使用的输入特征图名称列表。
- `MODEL.ROI_HEADS.IOU_THRESHOLDS`: RoI被视为背景或前景的IoU重叠阈值。
- `MODEL.ROI_HEADS.IOU_LABELS`: 每个IoU阈值对应的标签。
- `MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE`: 每个图像用于训练ROI heads的RoI minibatch大小。
- `MODEL.ROI_HEADS.POSITIVE_FRACTION`: 前景RoI的比例。
- `MODEL.ROI_HEADS.SCORE_THRESH_TEST`: 测试阶段使用的最小分数阈值。
- `MODEL.ROI_HEADS.NMS_THRESH_TEST`: 非最大抑制 (NMS) 阈值。
- `MODEL.ROI_HEADS.PROPOSAL_APPEND_GT`: 是否在采样RoIs之前使用ground-truth boxes扩充proposals。

# ROI box head ： 对感兴趣区域的分类和定位
- C.MODEL.ROI_BOX_HEAD.NAME：ROI Box Head 的名称，通常会根据具体的模型选择对应的 ROI Box Head 类型，比如 FastRCNNConvFCHead。

- C.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE：边界框回归损失函数的类型，常见的选项包括 "smooth_l1"、"giou"、"diou" 和 "ciou"，用于衡量预测边界框与真实边界框之间的差异。

- C.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT：边界框回归损失的权重，用于平衡边界框回归损失与其他损失之间的梯度大小。

- C.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS：用于归一化边界框回归目标的默认权重，这些权重被经验性地选择，以使得目标具有近似单位方差。

- C.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA：平滑L1损失与L2损失之间的过渡点。设置为0.0可使损失简单地成为L1损失。

- C.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION 和 C.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO：感兴趣区域（RoI）的特征池化配置，决定了对RoI执行特征提取的分辨率和采样比例。

- C.MODEL.ROI_BOX_HEAD.FC_DIM 和 C.MODEL.ROI_BOX_HEAD.CONV_DIM：RoI Box Head 中全连接层和卷积层的维度。

- C.MODEL.ROI_BOX_HEAD.NORM：卷积层的归一化方法，可选的归一化方法包括 "GN"（组归一化）和 "SyncBN"（同步批归一化）。

- C.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG：是否使用类别无关的边界框回归。

- C.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES：是否使用由RoI Box Head预测的边界框而不是提议框（proposal boxes）来训练RoI头。
# Cascaded Box Head：
- C.MODEL.ROI_BOX_CASCADE_HEAD.BBOX_REG_WEIGHTS：用于级联边界框（cascade bounding boxes）的回归权重，定义了级联阶段中每个阶段的边界框回归权重。在级联目标检测模型中，通常会通过级联多个阶段来逐步提升边界框预测的精度。

- C.MODEL.ROI_BOX_CASCADE_HEAD.IOUS：级联目标检测模型中每个阶段的 IoU（交并比）阈值，用于决定是否将预测的边界框传递到下一个级联阶段。

# Mask head
- NAME: 指定了 ROI Mask Head 的类型，这里是 "MaskRCNNConvUpsampleHead"。
- POOLER_RESOLUTION: RoI pooling 的输出分辨率。
- POOLER_SAMPLING_RATIO: RoI pooling 的采样比例。
- NUM_CONV: 在掩膜头部中使用的卷积层数量。
- CONV_DIM: 卷积层的通道数。
- NORM: 卷积层的归一化方法，可以选择 "" (无归一化), "GN" (组归一化), "SyncBN" (同步批归一化)。
- CLS_AGNOSTIC_MASK: 是否对掩膜预测使用类别无关模式。
- POOLER_TYPE: 应用于每个 RoI 的特征图的池化操作类型，这里是 "ROIAlignV2"。

# ROI Keypoint Hea
- NAME: 指定了 ROI Keypoint Head 的类型，这里是 "KRCNNConvDeconvUpsampleHead"。
- POOLER_RESOLUTION: RoI pooling 的输出分辨率。
- POOLER_SAMPLING_RATIO: RoI pooling 的采样比例。
- CONV_DIMS: 在头部中使用的卷积层的通道数，这里设置为一个长度为 8 的元组，每个元素为 512。
- NUM_KEYPOINTS: COCO 数据集中的关键点数量，这里是 17 个。
- MIN_KEYPOINTS_PER_IMAGE: 训练时排除关键点太少（或没有）的图像。
- NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS: 如果为 True，则在计算关键点损失时按可见关键点的总数归一化。
- LOSS_WEIGHT: 关键点损失的多任务权重。
- POOLER_TYPE: 应用于每个 RoI 的特征图的池化操作类型，这里是 "ROIAlignV2"。

## 语义分割头部

- `SEM_SEG_HEAD.NAME`: 指定了语义分割头部的类型为 "SemSegFPNHead"。
- `SEM_SEG_HEAD.IN_FEATURES`: 指定用于语义分割头部的输入特征图列表，这里包括了多个特征图 ["p2", "p3", "p4", "p5"]。
- `SEM_SEG_HEAD.IGNORE_VALUE`: 语义分割标签中被忽略的值，对应的像素不计算损失，这里设定为 255。
- `SEM_SEG_HEAD.NUM_CLASSES`: 语义分割任务中的类别数量，这里设定为 54 类。
- `SEM_SEG_HEAD.CONVS_DIM`: 语义-FPN 头部内部 3x3 卷积层的通道数，设定为 128。
- `SEM_SEG_HEAD.COMMON_STRIDE`: 语义-FPN 头部输出上采样到的公共步长，设定为 4。
- `SEM_SEG_HEAD.NORM`: 卷积层的归一化方法，这里使用了 "GN"（组归一化）。
- `SEM_SEG_HEAD.LOSS_WEIGHT`: 语义分割任务的损失权重，设定为 1.0。

## 全景分割输出结果合并

- `PANOPTIC_FPN.INSTANCE_LOSS_WEIGHT`: 所有实例检测/分割头部的损失权重，设定为 1.0。
- `PANOPTIC_FPN.COMBINE.OVERLAP_THRESH`: 全景分割输出结果合并时的重叠阈值，设定为 0.5。
- `PANOPTIC_FPN.COMBINE.STUFF_AREA_LIMIT`: 合并操作中用于定义杂项区域的面积限制，设定为 4096。
- `PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH`: 合并实例时的置信度阈值，设定为 0.5。

## RetinaNet 参数

- **NUM_CLASSES**: 前景类别数量，这里设定为 80 类。
- **IN_FEATURES**: RetinaNet 使用的特征图列表，包括了多个特征图 ["p3", "p4", "p5", "p6", "p7"]。
- **NUM_CONVS**: 分类和边界框 tower 中使用的卷积层数量，不包括最后用于 logits 的卷积层。
- **IOU_THRESHOLDS**: 用于标记 anchor 的 IoU 重叠阈值，设定为 [0.4, 0.5]。
- **IOU_LABELS**: 每个 anchor 的标签类型，设定为 [0, -1, 1]。
- **PRIOR_PROB**: 训练开始时用于设置分类器 subnet logits 层偏置的先验概率，用于处理类别不平衡问题。
- **SCORE_THRESH_TEST**: 推断时分类分数的阈值，只有得分 > 0.05 的 anchor 会被考虑用于推断。
- **TOPK_CANDIDATES_TEST**: NMS 之前保留的 topk 候选框数量，设定为 1000。
- **NMS_THRESH_TEST**: NMS 阈值，设定为 0.5。
- **BBOX_REG_WEIGHTS**: 用于归一化 RetinaNet 边界框回归目标的权重。
- **FOCAL_LOSS_GAMMA**: Focal 损失函数的 gamma 参数，设定为 2.0。
- **FOCAL_LOSS_ALPHA**: Focal 损失函数的 alpha 参数，设定为 0.25。
- **SMOOTH_L1_LOSS_BETA**: Smooth L1 损失函数的 beta 参数，设定为 0.1。
- **BBOX_REG_LOSS_TYPE**: 边界框回归损失的类型，这里使用 "smooth_l1"。
- **NORM**: 卷积层的归一化方法，这里为空字符串 ""。

## ResNe[X]t 参数

- **DEPTH**: ResNet 的深度，这里设定为 50。
- **OUT_FEATURES**: ResNet 输出的特征图名称列表，这里设定为 ["res4"]。
- **NUM_GROUPS**: ResNeXt 中使用的分组数量，这里为 1。
- **NORM**: 卷积层的归一化方法，这里使用 "FrozenBN"。
- **WIDTH_PER_GROUP**: 每个分组的基准宽度。
- **STRIDE_IN_1X1**: 是否将 stride 2 的卷积放置在 1x1 过滤器上，这里设定为 True。
- **RES5_DILATION**: 在 "res5" 阶段应用的空洞卷积率，设定为 1。
- **RES2_OUT_CHANNELS**: res2 阶段的输出通道数，设定为 256。
- **STEM_OUT_CHANNELS**: 网络的初始卷积阶段的输出通道数，设定为 64。
- **DEFORM_ON_PER_STAGE**: 是否在各个阶段应用可变形卷积。
- **DEFORM_MODULATED**: 是否使用调制可变形卷积。
- **DEFORM_NUM_GROUPS**: 可变形卷积中的分组数量。

## 学习率调度器

- **LR_SCHEDULER_NAME**: 学习率调度器的名称，可以选择 `WarmupMultiStepLR` 或 `WarmupCosineLR`。
- **MAX_ITER**: 训练的最大迭代次数。
- **BASE_LR**: 初始学习率。
- **BASE_LR_END**: 仅用于 `WarmupCosineLR`，表示结束时的学习率。
- **MOMENTUM**: 动量参数。
- **WEIGHT_DECAY**: 权重衰减参数，应用于参数归一化层的权重。
- **WEIGHT_DECAY_NORM**: 用于归一化层参数的额外权重衰减。
- **GAMMA**: 每经过一定迭代数降低学习率的倍数。
- **STEPS**: 定义学习率下降的迭代步骤。
- **NUM_DECAYS**: 在 `WarmupStepWithFixedGammaLR` 调度中学习率衰减的次数。
- **WARMUP_FACTOR**: 预热期间的学习率倍数。
- **WARMUP_ITERS**: 预热迭代次数。
- **WARMUP_METHOD**: 预热学习率的方法，可以选择 `linear` 等。
- **RESCALE_INTERVAL**: 预热后是否重新调整学习率计划的间隔。

## 检查点和批次

- **CHECKPOINT_PERIOD**: 每隔一定迭代周期保存一次模型检查点。
- **IMS_PER_BATCH**: 每个批次的图像数量，也是每次迭代的训练图像数量。

## 硬件和并行

- **REFERENCE_WORLD_SIZE**: 训练使用的 GPU 数量，用于自动调整相关配置。

## 偏置和学习率

- **BIAS_LR_FACTOR**: 偏置学习率的缩放因子。
- **WEIGHT_DECAY_BIAS**: 偏置参数的额外权重衰减。

## 梯度裁剪

- **CLIP_GRADIENTS**: 是否开启梯度裁剪。
  - **CLIP_TYPE**: 梯度裁剪的类型，可以选择 `value` 或 `norm`。
  - **CLIP_VALUE**: 最大的梯度绝对值。
  - **NORM_TYPE**: L-p 范数中的 p 值，用于 `norm` 类型的梯度裁剪。

## 自动混合精度训练

- **AMP.ENABLED**: 是否开启自动混合精度训练。

# 测试和评估参数

- **EXPECTED_RESULTS**: 用于端到端测试以验证预期精度的指标列表。每个条目是一个列表，包含 `[task, metric, value, tolerance]`，例如 `['bbox', 'AP', 38.5, 0.2]`。
- **EVAL_PERIOD**: 在训练过程中评估模型的周期（以步数为单位）。设置为 0 表示禁用评估。

- **KEYPOINT_OKS_SIGMAS**: 用于计算关键点 OKS 的 sigmas 列表。如果为空，则使用 COCO 中的默认值。否则，应该是与 `ROI_KEYPOINT_HEAD.NUM_KEYPOINTS` 长度相同的浮点数列表。
- **DETECTIONS_PER_IMAGE**: 推断期间每个图像返回的最大检测数。基于 COCO 数据集的限制，默认为 100。
- **AUG**: 控制测试阶段的数据增强设置。
  - **ENABLED**: 是否启用数据增强。
  - **MIN_SIZES**: 数据增强的最小尺寸列表。
  - **MAX_SIZE**: 数据增强的最大尺寸。
  - **FLIP**: 是否在测试期间进行水平翻转。
- **PRECISE_BN**: 控制测试阶段是否启用精确的批量归一化计算。
  - **ENABLED**: 是否启用精确的批量归一化。
  - **NUM_ITER**: 进行精确批量归一化计算的迭代次数。


# 其他

- OUTPUT_DIR: 输出文件的保存目录。
- SEED: 随机数种子，用于提高可重现性。
- CUDNN_BENCHMARK: 是否启用 cuDNN 加速优化。
- VIS_PERIOD: 训练时进行可视化的周期。