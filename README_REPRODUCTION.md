# BRIGHT 论文 UNet 基线复现指南

本文档面向论文：

> Chen et al., *BRIGHT: a globally distributed multimodal building damage
> assessment dataset with very-high-resolution for all-weather disaster
> response*, Earth System Science Data, 17, 6217–6253, 2025.

核查基准是官方仓库 `master` 的提交
`59269142f3a3550320513e362692732f46486985`。论文参数来自仓库历史提交
`75728ac` 中的 `BRIGHT_paper.pdf`，不是根据 README 猜测。本文档和烟测不修改
原始网络、标签、官方划分或评价公式。

## 1. 官方源码位置与执行链路

| 功能 | 官方文件 | 关键行为 |
|---|---|---|
| 训练入口 | `bda_benchmark/script/standard_ML/train_UNet.py` | 6 通道输入、4 类输出、AdamW、每 500 step 验证 |
| 推理入口 | `bda_benchmark/script/standard_ML/infer_UNet.py` | 直接前向和 argmax，输出原始/着色 PNG，并汇总 mIoU |
| 数据集 | `bda_benchmark/dataset/make_data_loader.py` | 光学 RGB + 复制为 3 通道的单波段 SAR + 单通道标签 |
| 数据增强 | `bda_benchmark/dataset/imutils.py` | 随机裁剪、水平/垂直翻转、90/180/270°旋转 |
| UNet | `bda_benchmark/model/UNet.py` | 5 级编码器，4 级转置卷积解码器，BN + ReLU |
| Lovász 损失 | `bda_benchmark/util_func/lovasz_loss.py` | 官方 Lovász-Softmax 实现，忽略标签值 255 |
| 指标 | `bda_benchmark/util_func/metrics.py` | 混淆矩阵、OA、逐类 IoU、mIoU、定位/损伤 F1 |
| 标准划分 | `bda_benchmark/dataset/splitname/standard_ML/` | 当前 14 事件 train/val/test 列表 |

训练和推理脚本顶部保留了作者机器上的绝对 `sys.path`。不要编辑这些文件；从
`bda_benchmark` 目录运行并显式设置 `PYTHONPATH` 即可。

数据流如下：

1. 灾前 TIFF 只取前 3 个波段；灾后 SAR TIFF 必须被读取为二维单波段数组。
2. SAR 波段复制三次，与 RGB 分别按 ImageNet 的三通道均值/标准差归一化。
3. 两者拼接为 `[B, 6, H, W]`，UNet 输出 `[B, 4, H, W]`。
4. 标签含义为 `0=background`、`1=intact`、`2=damaged`、`3=destroyed`；
   随机裁剪的填充值为 `255`。
5. 推理用四类 logits 的 argmax。`apply_tta_inference()` 虽已实现，但
   `run_inference()` 没有调用它，论文基线复现不应擅自开启 TTA。

## 2. 论文参数与当前代码对照

| 项目 | 论文正文 | 当前代码/命令 | 结论 |
|---|---|---|---|
| 优化器 | AdamW | AdamW | 一致 |
| 学习率 | `1e-4` | 默认及 README 均为 `1e-4` | 一致 |
| 权重衰减 | `5e-3` | Python 默认 `5e-3` | 一致 |
| batch size | 16 | Python 默认 8；README/集群脚本传 16 | 必须显式传 16 |
| 优化器更新次数 | 50,000 iterations | README 传 `--max_iters 800000` | 实际一致：该参数是样本数，`800000 / 16 = 50000` 个 batch/step |
| Python 默认训练量 | 未使用 | `240000 / 8 ≈ 30000` step | 默认值不能复现论文 |
| 损失 | `CE + Lovász` | `CE + 0.75 × Lovász` | 明确不一致；本文不修改公式 |
| crop | 论文只说明随机裁剪 | README/脚本传 640 | 640 是公开代码配置，论文正文未写尺寸 |
| 增强 | 翻转、90°倍数旋转、随机裁剪 | 水平/垂直翻转；旋转总是 90/180/270°；随机裁剪 | 大类一致，具体概率/是否允许 0°仅代码给出 |
| 模型选择 | 验证集调参，测试集最终报告 | 每 500 step 同时计算 val 和 test；仅按 val mIoU 存 best | 选择准则未用 test，但测试集被重复查看 |
| 随机性 | 未报告 seed | 代码未设 Python/NumPy/PyTorch seed | 无法保证逐位复现 |
| 学习率调度 | 未说明 | 无 scheduler | 按常数学习率运行 |

一个额外实现风险是 `F.cross_entropy(output_clf, labels_clf)` 没有设置
`ignore_index=255`，而随机裁剪在图像小于 crop 时会填充 255。官方 1024×1024
切片配 640 crop 通常不会触发填充；若使用更小样本训练，CE 会因类别 255 报错。
不要通过修改标签或损失来“修复”论文复现，应保持官方尺寸，或把实验标记为代码修复版。

### 论文报告值（仅作目标，不是本仓库本次实测）

| 模型 | F1loc | F1clf | OA | mIoU | IoU: background/intact/damaged/destroyed |
|---|---:|---:|---:|---:|---|
| UNet | 89.19 | 54.69 | 94.73 | 59.74 | 96.16 / 69.84 / 17.64 / 55.32 |

`F1loc` 是背景/建筑二分类 F1；`F1clf` 是仅在真实建筑像素上计算的 intact、
damaged、destroyed 三个 F1 的调和平均；OA、逐类 IoU 和 mIoU 使用完整四类图。

## 3. 论文划分与当前划分

论文描述按事件分别约 7:1:2 划分后合并。论文表 5 只对以下 9 个事件报告结果：

`bata-explosion`、`beirut-explosion`、`congo-volcano`、`haiti-earthquake`、
`hawaii-wildfire`、`la_palma-volcano`、`libya-flood`、`morocco-earthquake`、
`turkey-earthquake`。

论文中的 Ukraine、Myanmar、Mexico 因灾前光学影像不能再分发而显示为 `–`。
当前 `standard_ML` 列表还包含后来加入的 `noto-earthquake`、
`marshall-wildfire`，以及需要用户自行准备光学数据的 Ukraine、Myanmar、Mexico，
总数为 2917/449/877；直接使用它们不是论文表 4 的同一测试范围。

更重要的是，仓库最早公开的论文期列表（提交 `c0032ed`）有两个跨集合重复项：
`bata-explosion_00000049` 同时在 train/test，`libya-flood_00000024` 同时在
train/val。当前列表已用 `bata-explosion_00000050` 和
`libya-flood_00000034` 替换训练项，消除了泄漏；当前 9 事件子集计数为
2229/349/661，三个集合互不重叠。

因此推荐的主复现使用“当前官方无泄漏划分 + 论文 9 事件范围”，生成派生列表，
但绝不覆盖官方列表：

```bash
cd /path/to/BRIGHT
mkdir -p runs/essd-unet/splits
for split in train val test; do
  grep -Ev '^(marshall-wildfire|noto-earthquake|mexico-hurricane|myanmar-hurricane|ukraine-conflict)_' \
    "bda_benchmark/dataset/splitname/standard_ML/${split}_set.txt" \
    > "runs/essd-unet/splits/${split}_set.txt"
done
wc -l runs/essd-unet/splits/*.txt
cat runs/essd-unet/splits/*.txt | sort | uniq -d  # 应为空
```

若研究目的是审计历史结果，可从 Git 历史导出旧列表并明确标注“含泄漏”，不可把
它与推荐主结果混写：

```bash
git show c0032ed:dfc25_benchmark/dataset/splitname/train_setlevel.txt > runs/essd-unet/splits/train_historical.txt
git show c0032ed:dfc25_benchmark/dataset/splitname/val_setlevel.txt > runs/essd-unet/splits/val_historical.txt
git show c0032ed:dfc25_benchmark/dataset/splitname/holdout_setlevel.txt > runs/essd-unet/splits/test_historical.txt
```

这两个设置可能不会产生相同训练结果。论文没有提供足够信息判断内部训练最终使用了
泄漏修正前还是修正后的两项，因此应同时保存 commit、列表 checksum 和实验标签。

## 4. 数据与模型权重

官方入口：

- 数据集 Zenodo：<https://zenodo.org/records/14619797>
- 数据集 Hugging Face：<https://huggingface.co/datasets/Kullervo/BRIGHT>
- 官方模型权重 Zenodo：<https://zenodo.org/records/15349462>
- Ukraine/Myanmar/Mexico 光学影像准备：`tutorial.md`
- 手工配准参考：数据记录中的 `aoi_reference_kmz.zip`

下载后至少需要以下逐像素配对内容：

```text
${DATASET_ROOT}/
├── pre-event/
│   └── <sample>_pre_disaster.tif       # H×W×>=3，读取前 3 波段
├── post-event/
│   └── <sample>_post_disaster.tif      # H×W，单波段 SAR
└── target/
    └── <sample>_building_damage.tif    # H×W，整数 0/1/2/3
```

论文数据切片应为 1024×1024。训练前对每个列表项检查三文件均存在、空间尺寸一致、
SAR/标签是二维数组、标签唯一值属于 `{0,1,2,3}`。不要重采样标签、改变类别编号或
自动把缺失的灾前光学影像替换为别的数据。

模型记录应下载 UNet 对应的 `.pth` state dict，而不是其他网络或优化器 checkpoint。
当前 `infer_UNet.py` 要求 key 不带 `module.` 前缀，并且首层/末层应分别兼容
`[64, 6, 3, 3]` 和 `[4, 64, 1, 1]`。从 Zenodo 页面记录实际文件名、文件大小、
版本和 checksum；官方 README 没有把这些元数据写进仓库，不能凭名称猜测。

## 5. Linux GPU 环境

推荐 Python 3.10。PyTorch 2.2.1 和 torchvision 0.17.1 是匹配组合；以下示例
使用 CUDA 12.1 wheel，宿主机必须有足够新的 NVIDIA 驱动：

```bash
conda create -n bright-repro python=3.10 -y
conda activate bright-repro
python -m pip install --upgrade pip
python -m pip install torch==2.2.1 torchvision==0.17.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-reproduction.txt
python -m pip check
python - <<'PY'
import cv2, numpy, torch, torchvision
print("numpy", numpy.__version__)
print("torch", torch.__version__, "torchvision", torchvision.__version__)
print("cuda", torch.cuda.is_available(), torch.version.cuda)
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

CUDA 11.8 环境可把索引改为 `https://download.pytorch.org/whl/cu118`。不要同时安装
CPU wheel 和 CUDA wheel。

原始 `requirements.txt` 不能由严格解析器原样满足：`matplotlib==3.8.3` 声明
`numpy<2`，但文件同时固定 `numpy==2.2.1`；此外 PyTorch 2.2.1 的 NumPy bridge
也不适合 NumPy 2.x。`requirements-reproduction.txt` 只把 NumPy 调整为 1.26.4
并补上源码直接导入但原文件遗漏的 `opencv-python-headless`，没有改变模型行为。

## 6. CPU 最小烟测

烟测不会下载数据或权重，也不会调用硬编码 `.cuda()` 的官方入口。它临时创建一组
32×32 TIFF，调用原始 Dataset、UNet、Lovász 和 Evaluator，验证：

- TIFF 命名和读取、SAR 三通道复制、6 通道拼接；
- `[1,6,32,32] -> [1,4,32,32]` 前向与 argmax 推理；
- 代码实际使用的 `CE + 0.75×Lovász` 和一次 backward；
- 完美预测时定位 F1、三类损伤 F1、四类 mIoU 接近 1。

```bash
PYTHONDONTWRITEBYTECODE=1 python tools/smoke_test_unet.py
```

随机初始化模型的 OA/mIoU 只用于证明评价流程能执行，不是实验指标，不应与论文比较。

## 7. 论文对齐训练

从仓库根目录执行：

```bash
mkdir -p runs/essd-unet/checkpoints
cd bda_benchmark
PYTHONPATH="$PWD" python script/standard_ML/train_UNet.py \
  --dataset BRIGHT \
  --train_batch_size 16 \
  --eval_batch_size 4 \
  --num_workers 16 \
  --crop_size 640 \
  --max_iters 800000 \
  --learning_rate 1e-4 \
  --weight_decay 5e-3 \
  --model_type UNet \
  --model_param_path ../runs/essd-unet/checkpoints \
  --train_dataset_path "$DATASET_ROOT" \
  --train_data_list_path ../runs/essd-unet/splits/train_set.txt \
  --val_dataset_path "$DATASET_ROOT" \
  --val_data_list_path ../runs/essd-unet/splits/val_set.txt \
  --test_dataset_path "$DATASET_ROOT" \
  --test_data_list_path ../runs/essd-unet/splits/test_set.txt
```

资源不足时可以减少 `num_workers`，但改变 batch size、crop size 或
`max_iters / batch_size = 50000` 的优化器 step 数会形成不同实验。脚本没有 AMP、
梯度累积、scheduler 或断点中的优化器状态；添加这些都必须另记为非官方变体。

训练保存到带时间戳的目录，只有验证 mIoU 提升时写 `best_model.pth`。保存以下信息：
Git commit、GPU/驱动/CUDA、Python/PyTorch/NumPy 版本、三个列表 checksum、数据记录
版本/checksum、完整 stdout 和最终 checkpoint checksum。

## 8. 推理与评价

```bash
cd bda_benchmark
PYTHONPATH="$PWD" python script/standard_ML/infer_UNet.py \
  --model_path ../runs/essd-unet/checkpoints/BRIGHT/<run>/best_model.pth \
  --test_dataset_path "$DATASET_ROOT" \
  --test_data_list_path ../runs/essd-unet/splits/test_set.txt \
  --output_dir ../runs/essd-unet/inference
```

控制台的 Overall Metrics 给出四类 OA、mIoU 和逐类 IoU。训练脚本还计算论文表中的
F1loc 与 F1clf，但推理脚本当前没有打印它们；要对齐论文表 4，应保存训练脚本在最佳
validation round 对应的 test 输出，或写独立只读评价包装器调用同一个 `Evaluator`。
不要改变 `Evaluator` 公式，也不要把按事件 mIoU 的简单平均与全测试集混淆矩阵 mIoU
混为一项。

## 9. 常见错误

- `ModuleNotFoundError: dataset`：从 `bda_benchmark` 运行并设置
  `PYTHONPATH="$PWD"`，不要改作者绝对路径。
- 依赖解析报 NumPy 冲突：使用 `requirements-reproduction.txt`，不要安装 NumPy 2.x。
- `No module named cv2`：安装 `opencv-python-headless==4.11.0.86`。
- `Torch not compiled with CUDA enabled`：安装对应 CUDA wheel，并先确认
  `torch.cuda.is_available()`；CPU 只能运行烟测，官方入口硬编码 CUDA。
- CUDA OOM：先确认 GPU 显存是否能承担 batch 16、crop 640；降低 batch 会改变
  BatchNorm 和论文设置，不能仍宣称严格复现。
- `Target 255 is out of bounds`：输入切片小于 crop 触发了标签填充；论文数据应为
  1024×1024，不要在正式复现中用小图直接训练。
- `Expected 3-dimensional input` 或 shape mismatch：SAR TIFF 必须被读取为 H×W，
  不能是 H×W×1；三类 TIFF 尺寸必须一致。
- state-dict key 带 `module.`：下载了 DataParallel 或错误格式 checkpoint；先核对
  Zenodo 文件，不能静默丢弃 key。
- 某事件指标为 `nan`：列表中没有该事件时，事件 evaluator 的混淆矩阵为空；只比较
  实际包含样本的 9 个论文事件。
- `SyntaxWarning: "is" with a literal`：来自官方 `lovasz_loss.py` 中的历史写法；
  当前烟测仍可通过。严格基线复现保留该文件，代码清理应作为单独实验变更。

## 10. 当前尚未执行

当前云实例没有 BRIGHT 真数据、官方 UNet 权重或 NVIDIA GPU，因此未执行：完整数据
完整性/checksum 核验、50,000-step GPU 训练、官方 checkpoint 的全测试集推理，以及
论文表 4 的数值对比。只有 CPU 合成小样本烟测可以在本环境中验证；其结果不得作为
论文复现实验结果。
