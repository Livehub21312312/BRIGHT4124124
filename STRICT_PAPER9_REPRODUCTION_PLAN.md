# BRIGHT UNet 基线核对与 9 事件训练准备

> 状态：仅完成论文/源码审查、派生划分工具和训练烟测工具；未启动正式训练，也未在本仓库环境冒充执行 Windows 真实数据训练烟测。

## 1. 先明确版本口径

当前仓库没有工作树中的论文 PDF。正式论文应以 [ESSD 2025 HTML](https://essd.copernicus.org/articles/17/6217/2025/essd-17-6217-2025.html) 和 [PDF](https://essd.copernicus.org/articles/17/6217/2025/essd-17-6217-2025.pdf) 为准。仓库历史提交 `3b2f9c09d6de6c198d994a233c40d957037da9a0` 中曾包含 2025-01-09 生成的 `BRIGHT_paper.pdf`，但它是早期稿，不是 ESSD 最终版。

这里存在一个会改变复现目标的关键版本差异：

| 资料 | 数据范围 | 标准划分/评价范围 |
|---|---|---|
| ESSD 2025 正式版，Sect. 2.4、Table 5/6 | 14 个事件，4246 对影像 | 每事件 7:1:2 后合并；正式标准测试覆盖 14 个事件 |
| 2025-01 仓库历史 PDF | 前 12 个事件；文中另有 DFC 数据 | 12 事件按 7:1:2；Table 5 只给出其中 9 个事件的 event-level mIoU |
| 当前 `standard_ML` 列表 | 14 个事件 | train/val/test = 2918/450/878，总计 4246 |
| 从当前列表筛出的历史 9 事件 | Bata、Beirut、Goma/Congo、Haiti、Hawaii、La Palma、Libya、Morocco、Turkey | 2229/349/661，集合间重复为 0 |

因此，`2229/349/661` 可以作为一个有明确 ID、可重复的“历史 9 事件子集”实验，但不能称为 ESSD 2025 最终论文的完整标准划分。若最终目标是复现正式论文 Table 5，必须使用 14 事件的 `2918/450/878`。本文件沿用任务指定文件名中的 `PAPER9`，但后文统一将其标成“历史 9 事件实验”。

## 2. 论文、源码和本地实验对照

### 2.1 训练设置

| 项目 | ESSD 2025 正式论文 | 当前官方源码/README | 结论 |
|---|---|---|---|
| 模型 | UNet；Appendix D 给出 5 级编码器 64/128/256/512/1024 | `bda_benchmark/model/UNet.py`，6 通道输入、4 类输出，结构一致 | 一致 |
| 输入 | 灾前光学 + 灾后 SAR | 光学取前三波段；单波段 SAR 复制为 3 通道，再拼成 6 通道 | 论文未写到 SAR 复制细节，以代码为准 |
| 划分 | 14 事件各自按 7:1:2，再合并 | 当前列表 2918/450/878，合计 4246 | 与最终论文数据总数吻合；论文未逐个发布 ID 哈希 |
| 优化器 | AdamW | AdamW，使用 PyTorch 默认 betas/eps | 一致 |
| 学习率 | `1e-4` | `1e-4`，无 scheduler，恒定学习率 | 一致；论文未说明 scheduler，代码确认没有 |
| weight decay | `5e-3` | 参数默认 `5e-3` | 一致 |
| batch size | 16 | README 命令 16；脚本参数默认值是 8 | 按 README 才与论文一致 |
| 训练迭代 | 50000 iterations | README 传 `max_iters=800000` | 代码中的 `max_iters` 是样本数；`800000 / 16 = 50000` 次优化器更新，故命令与论文一致 |
| 损失 | 公式为 `CE + Lovasz` | 实际为 `CE + 0.75 * Lovasz` | 明确不一致；严格代码复现应保留 0.75，并记录为论文/代码差异 |
| crop | 只写 random cropping | README 指定 640×640 | crop 大小来自代码/README，不是论文正文明确值 |
| flip | random flipping | 水平和垂直各独立约 0.5 概率 | 代码给出具体实现 |
| rotation | 90° increments | 每个训练样本必选 90/180/270°之一，不会选 0° | 论文没有说明“必旋转”；属于实现细节 |
| crop 类别约束 | 未说明 | 最多尝试 10 次，目标最大类别比例 `<0.75`；代码 `len(cnt > 1)` 疑似本应写 `len(cnt) > 1` | 为忠于当前基线不修正；修正会改变采样分布 |
| 归一化 | 未说明 | 光学和复制后的 SAR 都用 ImageNet RGB mean/std | 实现细节 |
| 随机种子 | 未报告 | 未设置 | 原论文的精确随机轨迹不可恢复 |
| 验证/选模 | 验证集选超参和 checkpoint，测试集评估泛化 | 每 500 个 optimizer step 验证；按 val mIoU 保存 best | 选模条件一致 |
| 训练中测试 | 论文方法要求测试集只做泛化评估 | 训练脚本每 500 步也调用 `test()` 并记录 test 指标 | 严格方案不沿用该测试窥视行为；测试只在模型冻结后运行一次 |
| checkpoint | 未说明内部格式 | 只保存 `model.state_dict()`；`--resume` 只加载匹配键，不恢复 optimizer/步数/RNG | 不是严格断点续训 |

`max_iters` 的精确定义来自 `MultimodalDamageAssessmentDatset`：它先将 ID 列表重复并截断到 `max_iters` 个样本，DataLoader 再按 batch 组成更新。一般换算为：

```text
optimizer_updates = ceil(max_iters / train_batch_size)   # drop_last=False
```

官方参数 800000 可被 16 整除，正好是 50000 次更新。对 9 事件 train=2229，它相当于约 358.91 次数据集遍历；前 2018 个列表 ID 出现 359 次，其余 211 个出现 358 次，然后整体 shuffle。`max_iters=50000, batch=1` 不是等价替代：它只有 50000 次样本曝光，而不是 800000 次。

### 2.2 指标的真实实现

正式论文 Sect. 3.4 说明 F1 用于定位和损伤分类，OA/mIoU 用于最终损伤图，但没有展开代码中的聚合细节。当前 `bda_benchmark/util_func/metrics.py` 与 `train_UNet.py` 的实际口径为：

- 混淆矩阵行是真值、列是预测。
- `F1loc`：把预测类别 `1/2/3` 合并为 building=1，与二值定位标签计算像素 precision、recall 和 F1。
- `F1clf`：只在真值 building 像素上构造 4 类混淆矩阵，分别计算 intact、damaged、destroyed 的 F1，再取三者调和平均：`3 / sum(1 / class_f1)`；它不是 macro arithmetic mean。
- `OA`：最终 4 类图上全部像素的对角线和/总像素数。
- `mIoU`：最终 4 类图中 background、intact、damaged、destroyed 四个 IoU 的算术平均。
- 混淆矩阵生成时只接受真值 `[0, num_class)`，因此 255 真值被评价器屏蔽。

指标名称的意图与论文一致，精确公式以代码为操作定义。但论文未给出用于最终表格的源码 commit、预测文件或混淆矩阵，所以无法仅凭正文独立证明表格由当前 commit 逐字节生成。

官方 `infer_UNet.py` 对测试 TIFF 做整幅 1024×1024 前向传播，不做 crop、滑窗、resize 或 TTA；本地 661 样本评价与这一源码路径一致。ESSD Appendix E 的 TTA 属于 Sect. 4.5 的后处理实验，不应混入 Table 5 的原始 UNet 基线口径。

### 2.3 论文指标与当前本地结果

| 来源/范围 | F1loc | F1clf | OA | mIoU | 类别明细 |
|---|---:|---:|---:|---:|---|
| ESSD 2025 Table 5，UNet，14 事件标准测试 | 87.97% | 72.24% | 95.47% | 64.94% | IoU：bg 96.19、intact 71.27、damaged 39.13、destroyed 53.17 |
| 2025-01 历史 PDF Table 4，UNet | 89.19% | 54.69% | 94.73% | 59.74% | IoU：bg 96.16、intact 69.84、damaged 17.64、destroyed 55.32 |
| 用户报告的本地官方 `standard_ML` checkpoint，在 661 个历史 9 事件测试 ID 上 | 89.2954% | 63.1907% | 95.9160% | 63.8854% | F1：intact 90.1686、damaged 42.5901、destroyed 77.4873 |

这里不计算差值，原因有三：

1. ESSD 正式表是 14 事件/878 测试样本，本地结果是 9 事件/661 样本。
2. 历史 PDF 的 set-level Table 4 对应其 12 事件标准设置；“9 of 12”出现在 event-level Table 5，并不等于已经证明 Table 4 只用了 661 个 ID。
3. 本地类别明细是 per-class F1，论文表中类别明细是 per-class IoU，不能逐项比较。

### 2.4 官方 checkpoint 的适用范围和重叠风险

当前权重：

```text
文件：ckpt_UNet_standard_ML_split.pth
大小：124274886 bytes
MD5：4dc3dc6995a74faaf37d663e6d200037
Zenodo：https://zenodo.org/api/records/15349462/files/ckpt_UNet_standard_ML_split.pth/content
```

文件名、Zenodo 记录和当前 README 都把它称为 `standard_ML`，但 raw state dict 不含训练 ID、split 哈希、训练 commit、seed 或 optimizer 状态。当前 14 事件列表自身无跨集合重复，不能反推该权重实际使用的训练清单。因而：

- 没有证据表明已发生训练/测试重叠；
- 也无法排除 checkpoint 生成时使用了不同版本列表或不同训练范围；
- 其 661 样本结果只能写成“官方 standard_ML 权重在历史 9 事件测试子集上的评价”，不能写成严格独立的 9 事件训练结果。

## 3. 两条复现实验轨道

### 轨道 A：ESSD 2025 最终论文基线（推荐的最终目标）

- 直接使用当前官方 14 事件列表：2918/450/878。
- batch 16、crop 640、AdamW、lr `1e-4`、weight decay `5e-3`、50000 optimizer updates。
- 为复现“代码基线”，损失保留 `CE + 0.75 Lovasz`；报告中同时披露论文公式写作 `CE + Lovasz`。
- validation 每 500 updates；只按 validation mIoU 选模；测试集在 best checkpoint 冻结后只运行一次。

### 轨道 B：历史 9 事件严格隔离实验（当前任务）

- 从当前列表按固定 9 个事件筛选，保留原顺序，得到 2229/349/661。
- 不改官方列表；用 `tools/prepare_paper9_splits.py` 在 `outputs/paper9_splits` 生成派生列表及 SHA-256 manifest。
- 训练期间只读 2229 train；选模只读 349 val；冻结后只对 661 test 评价一次。
- 训练目标、模型、loss、优化器、crop 和 optimizer update 数与代码基线保持一致。
- 结果必须命名为“历史 9 事件实验”，不能作为 ESSD 最终版 Table 5 的完整复现。

## 4. Windows 环境

已经实测可用的本地核心环境是 Windows、RTX 3050 Ti Laptop 4GB、PyTorch `2.5.1+cu121`。既然真实数据推理已成功，不建议为贴合仓库 pin 而重装 PyTorch。

仓库 `requirements.txt` 固定 `torch==2.2.1`、`torchvision==0.17.1`，同时固定 `numpy==2.2.1`；后者对这组旧 PyTorch/部分二进制扩展存在兼容风险。当前复现辅助依赖建议沿用 `requirements-reproduction.txt`，torch/torchvision 单独保留已验证版本：

```powershell
Set-Location 'D:\BRIGHT-Reproduction\BRIGHT'
git switch repro/essd-unet-baseline
git pull origin repro/essd-unet-baseline

python -m pip install -r .\requirements-reproduction.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

训练前记录：

```powershell
git rev-parse HEAD
python --version
python -m pip freeze | Out-File -Encoding utf8 .\outputs\environment-pip-freeze.txt
nvidia-smi | Out-File -Encoding utf8 .\outputs\environment-nvidia-smi.txt
```

## 5. 先生成并核对派生 9 事件列表

```powershell
Set-Location 'D:\BRIGHT-Reproduction\BRIGHT'
python .\tools\prepare_paper9_splits.py `
  --output-dir '.\outputs\paper9_splits'
```

必须看到 `status: passed`、2229/349/661 和三个 overlap count 均为 0。工具只读官方列表，生成：

```text
outputs/paper9_splits/train_set_paper9.txt
outputs/paper9_splits/val_set_paper9.txt
outputs/paper9_splits/test_set_paper9.txt
outputs/paper9_splits/paper9_split_manifest.json
```

## 6. 真实数据训练烟测（现在应执行的下一步）

烟测复用未修改的官方 Dataset、augmentation、UNet 和 Lovasz，实现 2 次真实反向传播和 optimizer update；它只检查少量 train TIFF，完全不读取 val/test TIFF。

4GB 显存先测试 AMP、micro-batch 1、crop 640：

```powershell
Set-Location 'D:\BRIGHT-Reproduction\BRIGHT'
python .\tools\smoke_train_unet_paper9.py `
  --dataset-root 'D:\BRIGHT-Reproduction\data' `
  --device cuda `
  --steps 2 `
  --sample-limit 8 `
  --micro-batch-size 1 `
  --crop-size 640 `
  --seed 1337 `
  --amp `
  --report '.\outputs\paper9_train_smoke\training_smoke_report.json'
```

成功标准：report 为 `passed`，两个 total loss 都为有限数，输出为 `N×4×640×640`，且 tracked parameter 的最大绝对变化大于 0。AMP 是显存适配变体，不是官方 FP32。AMP 版通过后，可去掉 `--amp` 再做一次 FP32 烟测，确认 4GB 是否能容纳更接近官方数值精度的 micro-batch 1；CUDA OOM 只说明显存不足，不应降低 crop 或修改网络后仍称为完全复现。

烟测每个 micro-batch 都执行一次 update，目的只是证明训练链路可用；它不模拟正式实验的 batch 16 或 gradient accumulation，所得 loss 和速度不得作为正式训练结果。

本仓库云环境没有 torch/GPU/真实数据，故这里只做了脚本语法、划分计数和派生 manifest 检查，没有声称完成上述训练烟测。

## 7. 4GB 显存下哪些参数能改

| 调整 | 对复现口径的影响 |
|---|---|
| `eval_batch_size: 4 → 1` | 不改变逐像素预测/混淆矩阵，通常可视为等价的内存调整 |
| `num_workers: 16 → 0` | 不改变数据定义，但会改变随机增强的具体 RNG 调度和速度；官方未给 seed，本就无法 bitwise 复现 |
| AMP | 模型结构和目标不变，但数值精度改变；必须标“AMP 变体” |
| micro-batch 1 + gradient accumulation 16 | 每次 update 的样本数可保持 16，但 UNet 含 BatchNorm，BN 统计只看到 micro-batch 1，所以不等价于真实 batch 16 |
| batch size 16 → 1 且不累积 | 改变梯度和样本曝光/更新关系；不是完整复现 |
| crop 640 → 更小 | 改变上下文、类别分布和训练任务；不是完整复现 |
| 少于 50000 updates | 改变训练预算；只能称 smoke/缩减实验 |
| 修改 loss 权重、标签或网络 | 改变基线，禁止用于本复现 |

已知 640 推理峰值约 1010 MiB 不能用于推断训练显存；训练还需要保留激活、梯度和 AdamW 一/二阶状态。先以上述烟测报告的 `peak_allocated_bytes`/`peak_reserved_bytes` 为准。如果真实 batch 16 的 FP32 训练无法在 4GB 上运行，则完全遵守官方 batch 16 的实验需要更大显存 GPU；梯度累积方案只能作为明确标注的资源适配实验。

若确认接受 4GB 资源适配轨道，拟采用 crop 640、micro-batch 1、accumulation 16、50000 optimizer updates（仍为 800000 次样本曝光）、val batch 1、workers 0；每个 micro loss 除以 16 后再 backward，累积完成才 `optimizer.step()`。lr、weight decay、损失权重、模型和标签保持不变；只有在 FP32 烟测不适配时才启用 AMP。该配置必须披露 AMP 状态以及 BatchNorm 的 micro-batch 统计差异，名称使用“资源适配复现”，不能使用“完全复现”。

## 8. 正式训练命令草案（现在不要执行）

下列命令只用于展示官方等价的 50000-update 参数，需要能容纳 batch 16 的 GPU。为防止官方 `train_UNet.py` 在训练期间读取真实 test，暂时把它的 `test_*` 参数也指向 validation；脚本输出中所谓 test 此时其实是重复 validation，不能当测试结果。最终测试仍应使用独立评价脚本和真正 test manifest。

```powershell
$env:PYTHONPATH = (Resolve-Path '.\bda_benchmark').Path
python .\bda_benchmark\script\standard_ML\train_UNet.py `
  --dataset 'BRIGHT-paper9' `
  --train_batch_size 16 `
  --eval_batch_size 1 `
  --num_workers 0 `
  --crop_size 640 `
  --max_iters 800000 `
  --learning_rate 1e-4 `
  --weight_decay 5e-3 `
  --model_type 'UNet' `
  --model_param_path '.\outputs\paper9_training_reference' `
  --train_dataset_path 'D:\BRIGHT-Reproduction\data' `
  --train_data_list_path '.\outputs\paper9_splits\train_set_paper9.txt' `
  --val_dataset_path 'D:\BRIGHT-Reproduction\data' `
  --val_data_list_path '.\outputs\paper9_splits\val_set_paper9.txt' `
  --test_dataset_path 'D:\BRIGHT-Reproduction\data' `
  --test_data_list_path '.\outputs\paper9_splits\val_set_paper9.txt'
```

这仍不是推荐在 4GB 笔记本上直接启动的命令，因为：batch 16 很可能 OOM；脚本不支持 AMP/累积；没有 seed；resume 不恢复 optimizer/RNG；只保存模型权重。烟测通过并由人工确认采用“更大 GPU 的严格 batch 16”或“4GB 资源适配变体”后，再新增独立正式 trainer，不能悄悄改变这些条件。

## 9. 正式 trainer 的验收设计（待确认后实现）

独立 trainer 应导入而不修改官方 Dataset/UNet/Lovasz，并满足：

1. 显式以 `optimizer_step=50000` 为终止条件；若使用累积，另记 `micro_step` 和 `samples_seen`。
2. 只用 train 更新；每 500 optimizer steps 用 val；best 只按 val mIoU；训练过程中禁止打开 test TIFF。
3. 保存 `best_val_miou.pt` 和可续训的 `latest.pt`。后者至少含 model、optimizer、AMP scaler、optimizer step、micro step、seed、Python/NumPy/torch/CUDA RNG、数据顺序/采样器状态、配置和 split SHA-256。
4. 采用写临时文件后原子替换；保留 best、latest 和少量带 step 的周期快照，避免每 500 步无限累积。
5. final checkpoint 冻结后，使用现有 `tools/evaluate_unet_testset.py` 对相应 test manifest 只评价一次。
6. 报告是否 FP32、是否 AMP、真实 batch、micro-batch、accumulation、BN 行为、crop、loss 权重和代码 commit。

## 10. 时间和磁盘测量

不从 0.353 秒单次推理外推训练时间。烟测后：

- 丢弃 CUDA 首次初始化/前若干 warm-up；用 `torch.cuda.synchronize()` 包围至少 20–50 个 optimizer steps，记录中位数和 P90。
- 单独测一次完整 349 样本 validation 时间。
- 若每 500 步验证，总计 100 次 validation；粗略预算为 `50000 × median_step_seconds + 100 × validation_seconds`，再加 checkpoint I/O。
- 记录墙钟时间和 GPU 峰值，不能只使用 Python 未同步计时。

官方权重文件约 118.5 MiB。AdamW 完整续训文件还需模型权重及两份优化器 moment，通常至少约为纯权重的 3 倍，再加元数据；以首次实测文件大小为准。建议仅保留 best + latest + 最近两个周期快照，预计远低于现有 80GB 空间，但正式启动前仍用 `Get-PSDrive D` 和实际 checkpoint 大小复核。

## 11. 实验记录规范

每次正式实验建立不可复用的 run 目录，至少保存：

```text
config.json
git_commit.txt
paper9_split_manifest.json（或 14 事件三个列表的 SHA-256）
environment-pip-freeze.txt
environment-nvidia-smi.txt
train.jsonl
validation.jsonl
best_val_miou.pt
latest.pt
final_test/evaluation_report.json
```

建议预注册 seed（例如 1337），同时设置 Python `random`、NumPy、torch CPU/CUDA seed、`cudnn.deterministic=True`、`cudnn.benchmark=False`。这提高新实验的可重复性，但原论文没有报告 seed，故仍不能恢复作者的精确随机轨迹。若资源允许，应报告多个 seed 的分布，而不是挑选 test 最优 seed。

## 12. 尚未解决、需人工确认

- 最终目标究竟是 ESSD 正式版 14 事件 Table 5，还是单独研究历史 9 事件；两者应分别命名、分别训练。
- 官方 checkpoint 的实际训练 ID、split 哈希、seed、源码 commit 和是否使用 AMP 未公开。
- 论文 `CE + Lovasz` 与代码 `CE + 0.75 Lovasz` 的差异无法从现有材料消解。
- 论文没有明确 crop 大小、精确增强概率、验证频率、初始化、scheduler（代码无 scheduler）和随机种子。
- 历史论文模型究竟对应哪一版修订列表，没有可验证的训练 manifest。
- RTX 3050 Ti 4GB 上 FP32 micro-batch 1 和 AMP micro-batch 1 的真实训练峰值尚待本地烟测。
- 在知道烟测显存/速度前，不确定采用更大 GPU 的真实 batch 16，还是接受带 AMP/梯度累积及 BN 差异的资源适配实验。
