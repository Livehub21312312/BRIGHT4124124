# BRIGHT 官方 UNet 权重：Windows 单样本推理

本阶段只对一个真实样本（默认取当前官方测试列表中的第一个论文 9 事件样本）执行一次
推理，保存原始类别图、彩色图和诊断报告。不会开始训练、反向传播、参数更新或完整
测试集评价；官方 Dataset、UNet、Evaluator、标签和划分文件均保持原样。

用户已经在 Windows 上确认无权重烟测通过：RTX 3050 Ti Laptop GPU（4 GB）、
PyTorch 2.5.1+cu121、输入 `[1,6,640,640]`、输出 `[1,4,640,640]`，CUDA 峰值约
1010 MiB。下面的命令是下一步官方权重推理，结果应以新生成的
`inference_report.json` 为准。

## 1. 官方 checkpoint 核验

2026-10-20 重新查询官方 Zenodo API，并下载实际文件核验得到：

| 项目 | 值 |
|---|---|
| Zenodo 记录 | <https://zenodo.org/records/15349462> |
| 文件名 | `ckpt_UNet_standard_ML_split.pth` |
| 直接下载 | <https://zenodo.org/api/records/15349462/files/ckpt_UNet_standard_ML_split.pth/content> |
| 文件大小 | `124274886` bytes（约 118.52 MiB） |
| MD5 | `4dc3dc6995a74faaf37d663e6d200037` |
| 格式 | 136 项 PyTorch `OrderedDict` state dict |
| 输入层权重 | `encoder1.block.0.weight`: `[64,6,3,3]` |
| 输出层权重 | `final_conv.weight`: `[4,64,1,1]` |

该文件已在当前官方 `UNet(in_channels=6, num_classes=4)` 上通过 `strict=True` 加载。

### PowerShell 下载与强制校验

```powershell
Set-Location 'D:\BRIGHT-Reproduction\BRIGHT'
New-Item -ItemType Directory -Force '.\checkpoints' | Out-Null

$CheckpointPath = '.\checkpoints\ckpt_UNet_standard_ML_split.pth'
$CheckpointUrl = 'https://zenodo.org/api/records/15349462/files/ckpt_UNet_standard_ML_split.pth/content'
Invoke-WebRequest -Uri $CheckpointUrl -OutFile $CheckpointPath

$ExpectedSize = 124274886
$ExpectedMD5 = '4dc3dc6995a74faaf37d663e6d200037'
$ActualSize = (Get-Item $CheckpointPath).Length
$ActualMD5 = (Get-FileHash $CheckpointPath -Algorithm MD5).Hash.ToLowerInvariant()

if ($ActualSize -ne $ExpectedSize) {
  throw "Checkpoint size mismatch: $ActualSize (expected $ExpectedSize)"
}
if ($ActualMD5 -ne $ExpectedMD5) {
  throw "Checkpoint MD5 mismatch: $ActualMD5 (expected $ExpectedMD5)"
}
Write-Host "Checkpoint verified: $ActualSize bytes, MD5 $ActualMD5"
```

不要在大小或 MD5 不一致时继续加载，也不要关闭 TLS 校验或用其他同名文件替换。
`tools/infer_one_real_sample.py` 会再次流式计算大小和 MD5，随后才调用
`model.load_state_dict(state_dict, strict=True)`。

## 2. 单个真实测试样本推理

先同步分支并确认 CUDA 环境：

```powershell
Set-Location 'D:\BRIGHT-Reproduction\BRIGHT'
git fetch origin
git switch repro/essd-unet-baseline
git pull --ff-only origin repro/essd-unet-baseline

python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

推荐先使用 640×640 中心裁剪，与已通过的真实数据烟测和官方训练 crop 一致。这里显式
选择官方测试列表中的 `bata-explosion_00000007`，不会遍历其余测试样本：

```powershell
python .\tools\infer_one_real_sample.py `
  --dataset-root 'D:\BRIGHT-Reproduction\data' `
  --checkpoint '.\checkpoints\ckpt_UNet_standard_ML_split.pth' `
  --split-file '.\bda_benchmark\dataset\splitname\standard_ML\test_set.txt' `
  --sample-id 'bata-explosion_00000007' `
  --device cuda `
  --forward-size 640 `
  --output-dir '.\outputs\official_unet_one_sample'
$LASTEXITCODE
```

不传 `--sample-id` 时，脚本只在指定列表中查找第一个 TIFF 三元组完整的论文 9 事件
样本；当前测试列表的第一个候选同样是 `bata-explosion_00000007`。传
`--forward-size 0` 可尝试整幅 1024×1024，但 4 GB GPU 没有必要在本阶段冒险；完整
影像和 640 中心裁剪的结果不可混作同一种输入设置。

输出目录包含：

- `<sample>_prediction_label.png`：单通道 `uint8`，像素值严格为预测类别 `0/1/2/3`；
- `<sample>_prediction_color.png`：使用官方推理脚本中的实际颜色表生成的 RGB 图；
- `inference_report.json`：checkpoint 校验、strict load、样本路径、原始 TIFF 元数据、
  crop 坐标、输入/输出尺寸、实际输出类别、设备、总显存、CUDA 峰值显存、推理时间、
  环境版本、混淆矩阵和单样本诊断指标。

颜色表与官方 `infer_UNet.py` 的代码一致：背景白色 `(255,255,255)`、intact 绿色
`(70,181,121)`、damaged `(228,189,139)`、destroyed `(182,70,69)`。官方代码中
class 0 旁的注释写着 black，但实际元组是白色；本包装器遵循代码元组，不根据注释改色。

脚本使用 `model.eval()`、`model.requires_grad_(False)` 和 `torch.inference_mode()`，没有
loss、optimizer、backward 或参数更新。它保存 PNG 后还会重新读取，确认原始类别值和
彩色图尺寸没有在编码时改变。

## 3. 官方 Evaluator 的真实计算方式

以下定义来自 `bda_benchmark/util_func/metrics.py` 和
`bda_benchmark/script/standard_ML/train_UNet.py` 的实际接线，不是根据名称推测。
混淆矩阵的行是真值、列是预测，真值不在 `[0,num_class)` 的像素会被过滤；UNet argmax
预测范围为 `0..3`。

### F1loc

- 真值：分类标签 `0/1/2/3` 中所有非零类别均映射为建筑类 `1`，背景为 `0`；源码先
  复制标签，再把 `2` 和 `3` 改为 `1`，原来的 `1` 保持不变。
- 预测：四类 argmax 中所有 `>0` 的值映射为建筑类 `1`。
- 用二分类混淆矩阵计算：
  `precision = TP / (FP + TP + 1e-7)`、
  `recall = TP / (FN + TP + 1e-7)`、
  `F1loc = 2PR / (P+R)`。
- 当样本中完全没有正例且没有正预测时，最后一步可能得到非有限值；JSON 用 `null`
  明确表示，不擅自改变公式。

### F1clf

- 只保留真值定位标签 `labels_loc > 0` 的像素，即真实建筑区域；真实背景像素不参与。
- 在这部分像素上建立四类混淆矩阵，但 `Damage_F1_score()` 排除 class 0，只返回
  class 1/intact、class 2/damaged、class 3/destroyed 的三个 F1。
- 每类使用 `TP/(TP+FP+1e-7)`、`TP/(TP+FN+1e-7)` 和带 `1e-7` 的 F1 分母。
- 训练脚本报告的 F1clf 不是宏平均，而是三个逐类 F1 的调和平均：
  `3 / (1/F1_intact + 1/F1_damaged + 1/F1_destroyed)`。任一类 F1 为零会使调和平均为零。

### OA、IoU 与 mIoU

- 在所有像素上使用完整 `0/1/2/3` 四类混淆矩阵。
- `OA = sum(diagonal) / sum(confusion_matrix)`。
- 每类 `IoU = TP / (GT_count + prediction_count - TP + 1e-7)`。
- `mIoU = np.nanmean([IoU0, IoU1, IoU2, IoU3])`。由于分母加了 `1e-7`，单样本中
  完全缺席的类别通常得到 0，而不是 NaN，因此仍会进入四类平均。

新增脚本直接实例化三个官方 `Evaluator` 并复用以上调用顺序，不改评价公式。报告中的
数值范围是 `0..1`，没有乘 100。

单个 640 crop 的这些数值只能说明推理与评价代码能够执行，尤其 absent class 会显著
影响 F1clf/mIoU；它们不是完整测试集指标、不是论文指标，也不能用于和论文表格比较。

## 4. 标准权重与论文 9 事件子集的边界

`ckpt_UNet_standard_ML_split.pth` 的架构与当前 UNet 完全兼容，但 Zenodo 没有随权重
提供训练样本 ID、split checksum 或“只使用 9 事件”的声明。该权重在 2025-05-07
加入记录，而官方仓库 2025-05-05 的 `standard_ML` 列表已经是 14 事件、
2918/450/878。

因此本次单样本即使来自论文 9 事件之一，也只能描述为“官方 standard_ML checkpoint
在一个 9 事件范围样本上的推理”。不得写成“论文 9 事件子集训练权重”或用它的结果
替代按 2229/349/661 列表重新训练得到的严格复现结果。Zenodo 中按事件命名的
`ckpt_UNet_cross_event_zeroshot_*.pth` 属于跨事件设置，也不等价于 9 事件标准划分。

请先运行上述一个样本，将 `inference_report.json`、原始标签 PNG 和彩色 PNG 保留。
确认 checkpoint、设备、输出类别和图像均正常后，再决定是否进入后续 GPU 训练配置；
本阶段不要调用官方完整 `infer_UNet.py` 遍历测试集。
