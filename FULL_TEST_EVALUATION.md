# BRIGHT 官方 UNet：完整测试集评价准备

本阶段只准备评价代码和 Windows 命令，不执行 661 样本正式评价，也不开始训练。新增
脚本直接使用官方 Dataset、UNet 和 Evaluator；官方模型、标签、划分及评价公式均未
修改。

## 1. 官方完整影像推理方式

核对 `bda_benchmark/dataset/make_data_loader.py` 与
`bda_benchmark/script/standard_ML/infer_UNet.py` 后，官方测试路径是：

1. `MultimodalDamageAssessmentDatset(..., crop_size=1024, type='test')`；只有 `type`
   包含 `train` 时才执行随机裁剪/翻转/旋转，测试模式只做通道处理和归一化。
2. 每个灾前 TIFF 取前三个光学波段；二维灾后 SAR 复制为三通道，二者分别归一化。
3. 官方推理 DataLoader 的 batch size 是 1。光学和 SAR 拼为
   `[1,6,1024,1024]`，UNet 直接输出 `[1,4,1024,1024]`。
4. 对四类 logits 直接 `argmax`，得到完整的 1024×1024 预测；直接保存原始类别 PNG
   和彩色 PNG。
5. 没有滑窗、重叠融合、裁剪拼接、resize、padding 恢复或输出后处理。UNet 卷积使用
   padding，1024 又能被四次池化的 16 整除，所以输出天然与原图同尺寸。
6. `apply_tta_inference()` 虽然定义了 6 路 TTA，但正式 `run_inference()` 没有调用，
   官方基线评价不应启用。

因此此前成功的 640×640 中心裁剪只证明设备和单样本链路可用，不能用于完整测试集
指标。`tools/evaluate_unet_testset.py` 对正式评价固定整幅 1024 输入；发生显存不足时
应关闭其他 GPU 程序或改用 CPU，不能静默改成 640 crop 或滑窗后仍称为官方设置。

## 2. 论文 9 事件测试子集

当前官方 `standard_ML/test_set.txt` 有 878 个非空且唯一的 ID。脚本通过精确前缀
`<event>_` 保留以下 9 个事件，而不是只按文件数量或模糊子串筛选：

| 事件 | 测试样本数 |
|---|---:|
| bata-explosion | 23 |
| beirut-explosion | 27 |
| congo-volcano | 27 |
| haiti-earthquake | 15 |
| hawaii-wildfire | 14 |
| la_palma-volcano | 189 |
| libya-flood | 26 |
| morocco-earthquake | 116 |
| turkey-earthquake | 224 |
| **合计** | **661** |

被排除的是 `marshall-wildfire`、`noto-earthquake`、`mexico-hurricane`、
`myanmar-hurricane`、`ukraine-conflict`，共 217 个。当前提交下：

- 官方源 test list SHA-256：
  `a118fcad3c740b7631e6282ae516b485e16286e383b348898de2c0698f5dfb6a`
- 按上述顺序生成、每行带换行的 661 ID 清单 SHA-256：
  `e7b4274da0d9737e7f658f24d38bd6fd459b4fd1041382b298c4040213caeff7`

脚本的 `--paper-nine-only` 会强制要求筛选结果恰好为 661，否则在加载数据和权重前
终止。它不会覆盖官方 `test_set.txt`，而是在输出目录生成
`evaluated_samples.txt`。

## 3. 官方评价逻辑

这里需要区分两个官方入口：

- `infer_UNet.py` 当前实际只把完整四类标签加入总 Evaluator，打印 OA、mIoU 和 IoU；
  F1clf 相关代码被注释掉。
- 论文所需的 F1loc/F1clf 接线存在于 `train_UNet.py` 的 `validation()` 和 `test()`。

新增脚本组合的是官方整图推理路径和 `train_UNet.py::test()` 的评价接线，并直接实例化
`bda_benchmark/util_func/metrics.py` 中的三个 `Evaluator`，没有另写指标公式：

- F1loc：真值 `1/2/3 → 1`，预测四类 argmax 的 `>0 → 1`，调用二类
  `Evaluator.Pixel_F1_score()`。
- 逐类分类 F1：只取真值建筑像素 `labels_loc > 0`，调用
  `Evaluator.Damage_F1_score()`。该方法明确排除 class 0，只返回 intact、damaged、
  destroyed 三类 F1；官方没有定义 background F1，报告中保持 `null`，不会自行补算。
- F1clf：严格按训练脚本使用三个损伤 F1 的调和平均
  `3 / sum(1 / damage_f1)`，不是宏平均。
- OA、四类 IoU 和 mIoU：把所有像素加入四类 Evaluator，分别调用
  `Pixel_Accuracy()`、`Intersection_over_Union()` 和
  `Mean_Intersection_over_Union()`。
- 三个混淆矩阵都完整写入 JSON，行是真值、列是预测；指标保存为 `0..1` 小数，不乘
  100。

指标是在全部成功样本累计后的混淆矩阵上一次性计算，不是先算单图指标再平均。若有
失败样本，状态为 `partial`、进程退出码为 1，JSON 会明确说明指标只覆盖成功样本，
不得当作完整 661 结果。

## 4. 先只审计清单

同步当前分支后，可以先运行不读 TIFF、不加载 checkpoint、不做推理的准备模式：

```powershell
Set-Location 'D:\BRIGHT-Reproduction\BRIGHT'
git fetch origin
git switch repro/essd-unet-baseline
git pull --ff-only origin repro/essd-unet-baseline

python .\tools\evaluate_unet_testset.py `
  --dataset-root 'D:\BRIGHT-Reproduction\data' `
  --test-list '.\bda_benchmark\dataset\splitname\standard_ML\test_set.txt' `
  --paper-nine-only `
  --output-dir '.\outputs\official_unet_paper9_test' `
  --prepare-only
```

检查：

```powershell
(Get-Content '.\outputs\official_unet_paper9_test\evaluated_samples.txt').Count
(Get-FileHash '.\outputs\official_unet_paper9_test\evaluated_samples.txt' -Algorithm SHA256).Hash.ToLowerInvariant()
```

预期为 661 和上面的派生清单 SHA-256。

## 5. 正式 661 样本评价命令（等待确认后再运行）

当前 checkpoint：

- `checkpoints\ckpt_UNet_standard_ML_split.pth`
- MD5 `4dc3dc6995a74faaf37d663e6d200037`
- 官方记录 <https://zenodo.org/records/15349462>

确认后执行：

```powershell
Set-Location 'D:\BRIGHT-Reproduction\BRIGHT'
python .\tools\evaluate_unet_testset.py `
  --dataset-root 'D:\BRIGHT-Reproduction\data' `
  --checkpoint '.\checkpoints\ckpt_UNet_standard_ML_split.pth' `
  --expected-checkpoint-md5 '4dc3dc6995a74faaf37d663e6d200037' `
  --test-list '.\bda_benchmark\dataset\splitname\standard_ML\test_set.txt' `
  --paper-nine-only `
  --device cuda `
  --expected-size 1024 `
  --progress-every 10 `
  --output-dir '.\outputs\official_unet_paper9_test'
$LASTEXITCODE
```

脚本固定 `model.eval()`、`torch.inference_mode()`、float32、batch size 1、无 TTA，
并以 `strict=True` 加载原始 state dict。它会重新计算 checkpoint MD5；不执行 loss、
backward、optimizer 或任何参数更新。

RTX 3050 Ti 的 640 烟测峰值约 1010 MiB，但 1024 整图峰值不能从该数字线性保证。
若遇到 OOM，先关闭浏览器、游戏覆盖层等 GPU 程序后重试；仍失败就使用
`--device cpu`。不要降低 `--expected-size`，该参数是输入断言而不是 resize 选项。

## 6. 输出文件

```text
outputs/official_unet_paper9_test/
├── evaluated_samples.txt       # 计划评价的精确 661 ID
├── completed_samples.txt       # 实际成功的 ID
├── failed_samples.csv          # 即使无失败也保留表头
├── evaluation_report.json      # 参数、环境、权重、计数、指标、混淆矩阵、失败清单
├── original/
│   └── <sample>_building_damage.png   # uint8 0/1/2/3
└── colored/
    └── <sample>_building_damage.png   # 官方颜色表 RGB
```

报告记录：源列表与派生列表 SHA-256、所有 CLI 设置、checkpoint 文件大小/MD5、是否
匹配已发布 standard_ML 文件、strict load 状态、PyTorch/CUDA、设备、峰值显存、总
耗时、累计输出类别、选择/成功/失败样本数、逐事件计数、F1loc、F1clf、三类损伤 F1、
四类 IoU、OA、mIoU 和三个混淆矩阵。

普通读取或单样本推理错误会写入 `failed_samples.csv` 后继续；CUDA OOM 会立即终止，
避免对剩余数百张图重复失败。只有 `status: passed`、`completed_samples: 661`、
`failed_samples: 0` 和退出码 0 同时满足，才是完整执行。

## 7. checkpoint 训练范围限制

`ckpt_UNet_standard_ML_split.pth` 的文件身份、架构和 strict load 已证实，但 Zenodo
没有提供它实际训练时的样本 ID 清单或 split checksum。该权重发布前，仓库中的
standard_ML 列表已经扩展为 14 事件。因此即使本次只在 661 个论文事件样本上测试，
报告也只能写成：

> 官方 standard_ML checkpoint 在论文 9 事件范围的当前无重复测试子集上的评价。

不能写成“由 2229 个论文 9 事件训练样本训练出的模型”，也不能直接把它的指标当作
严格论文复现结果。严格比较仍需后续使用明确保存的 2229/349/661 清单重新训练，并
记录权重与划分 checksum；本阶段不启动该训练。
