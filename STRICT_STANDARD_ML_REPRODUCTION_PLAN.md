# BRIGHT ESSD 2025 UNet standard_ML 正式复现计划

> 正式实验主线：ESSD 2025 最终论文的 14 事件 `standard_ML` 划分，train/val/test = 2918/450/878。历史 9 事件 2229/349/661 仅作为辅助实验保留，使用独立脚本、目录和报告，不与本实验合并。
>
> 当前阶段：代码审查、协议固定和真实训练烟测准备。未启动完整训练，未运行新的完整测试集评价，也未修改官方模型、Dataset、Evaluator、标签或标准划分。

## 1. 复现目标和资料依据

正式依据是 [ESSD 2025 论文](https://essd.copernicus.org/articles/17/6217/2025/essd-17-6217-2025.html)、当前仓库 README 和 `bda_benchmark` 实现。当前工作树没有论文 PDF；可访问的正式 [PDF 链接](https://essd.copernicus.org/articles/17/6217/2025/essd-17-6217-2025.pdf) 已用于核对。仓库历史中的 2025-01 `BRIGHT_paper.pdf` 是早期稿，不作为正式指标依据。

正式目标指标来自 ESSD Table 5 的 UNet 行：

| F1loc | F1clf | OA | mIoU | Background IoU | Intact IoU | Damaged IoU | Destroyed IoU |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 87.97% | 72.24% | 95.47% | 64.94% | 96.19% | 71.27% | 39.13% | 53.17% |

只有使用 14 事件训练/验证/测试范围、相同指标实现和明确独立的 test 才能进行正式对比。当前用户已经得到的 661 样本结果继续保存在 9 事件辅助轨道中，不计算它与上表的差值。

## 2. 数据边界

唯一正式划分来源：

```text
bda_benchmark/dataset/splitname/standard_ML/train_set.txt  2918
bda_benchmark/dataset/splitname/standard_ML/val_set.txt     450
bda_benchmark/dataset/splitname/standard_ML/test_set.txt    878
```

三个列表合计 4246 个样本、14 个事件，与 ESSD 最终版一致。此前数据完整性检查已经确认缺失引用、样本错误、警告和跨集合重复均为 0。

访问规则：

1. 训练只接收官方 `train_set.txt` 中的 ID。
2. validation 和 checkpoint 选择只接收官方 `val_set.txt` 中的 ID。
3. 训练、烟测、调参、选择 seed 和选择 checkpoint 时禁止读取或解析 `test_set.txt`，也不计算 test 哈希。
4. best checkpoint 和全部超参数冻结后，才由独立评价脚本读取官方 `test_set.txt`，且正式 test 只运行一次。
5. 数据根目录虽然同时包含所有 TIFF，Dataset 只能由当前阶段允许的 ID 列表索引，不能扫描后选择测试样本。

新增的 `tools/smoke_train_unet_standard_ml.py` 固定使用仓库中的官方 train/val 路径；它没有 test-list 参数，也不打开 test list。validation list 在烟测中只做 ID 数量、事件和 train/val 交集检查，validation TIFF 也不会读取。

## 3. 论文、官方代码和本地适配核对表

| 项目 | ESSD 2025 明确内容 | 官方 README/源码实际实现 | 4GB 本地方案及差异 |
|---|---|---|---|
| 数据范围 | 14 事件，每事件 7:1:2 后合并 | 当前列表 2918/450/878 | 保留全部 14 事件，不缩减为 9 事件 |
| 输入 | 灾前光学 + 灾后 SAR | 光学前三波段；单波段 SAR 复制为 3 通道；拼接成 6 通道 | 不变 |
| 模型 | UNet，五级编码器 64/128/256/512/1024 | `UNet(in_channels=6, num_classes=4)`，含 BatchNorm | 不变 |
| 标签 | background/intact/damaged/destroyed | 0/1/2/3 | 不变 |
| optimizer | AdamW | AdamW，PyTorch 默认 betas/eps | 不变 |
| learning rate | `1e-4` | 恒定 `1e-4`，没有 scheduler | 不变 |
| weight decay | `5e-3` | `5e-3` | 不变 |
| loss | 公式写 `CE + Lovasz` | `CE + 0.75 * Lovasz` | 以官方可执行代码为基线保留 0.75，同时披露论文/代码差异 |
| batch size | 16 | README 传 16；脚本默认值 8 | smoke 为 1；若正式用 micro-batch 1 + accumulation 16，BatchNorm 统计不同，不能称完全一致 |
| iterations | 50000 | README 传 `max_iters=800000` | 正式目标仍为 50000 optimizer updates/800000 样本曝光；smoke 只有 2 步 |
| crop | 只说明 random cropping | README 传 640；代码输出 640×640 crop | 保留 640 |
| flip | random flipping | 水平、垂直分别约 0.5 概率 | 不变 |
| rotation | 90° increments | 每个训练样本必选 90/180/270°之一，不含 0° | 不变，并记录这一代码细节 |
| crop 筛选 | 未说明 | 最多尝试 10 次；代码中 `len(cnt > 1)` 疑似实现错误 | 不修正，修正会改变训练分布 |
| normalization | 未说明 | 光学和复制 SAR 都用 ImageNet RGB mean/std | 不变 |
| validation 频率 | 未说明 | 每 500 optimizer updates | 正式方案优先保留；smoke 不跑 validation |
| best checkpoint | validation 选 checkpoint | 按 validation mIoU 保存 `best_model.pth` | 仍只按 val mIoU；禁止 test 参与 |
| 训练中 test | test 用于最终泛化评估 | 官方 `train_UNet.py` 每 500 步调用 `test()` | 官方脚本不直接用于严格训练；独立 trainer 必须完全没有 test 输入 |
| checkpoint 内容 | 未说明 | 只保存 model state；resume 不恢复 optimizer、步数或 RNG | 正式适配 trainer 需保存完整续训状态；属于工程增强，不改变训练目标 |
| seed | 未报告 | 未设置 | 预注册 1337；这是新增可重复性设置，不是论文参数 |
| precision | 未报告，官方代码没有 autocast | FP32 | 4GB 可能使用 AMP，必须标记为资源适配差异 |
| workers | 未报告 | README 为 16 | Windows 先用 0；会影响速度和具体随机流，但不改变样本集合 |
| eval batch | 未报告 | README 为 4 | 改为 1；模型处于 eval，聚合混淆矩阵不变 |

## 4. `max_iters` 的真实含义

`MultimodalDamageAssessmentDatset` 会重复官方训练 ID 并截断到 `max_iters` 个元素；DataLoader 才按 batch 组成 optimizer update。因此：

```text
optimizer_updates = ceil(max_iters / train_batch_size)
```

官方 README 参数为：

```text
max_iters = 800000 samples
train_batch_size = 16
optimizer_updates = 50000
```

对 2918 个训练 ID，800000 次样本曝光约为 274.16 次数据集遍历：列表前 468 个 ID 在截断前出现 275 次，其余 2450 个出现 274 次，之后 DataLoader 对展开列表整体 shuffle。

将 batch 改成 1 而仍传 `max_iters=800000` 会产生 800000 次 optimizer update，并非论文训练预算。资源适配 trainer 必须显式区分 `micro_step`、`optimizer_step` 和 `samples_seen`；若 micro-batch 1、accumulation 16，则 800000 micro steps 对应 50000 optimizer steps。

## 5. 官方 validation 和 checkpoint 逻辑的风险

官方 `train_UNet.py` 的实际流程：

1. 每个 batch 计算 `CE + 0.75 Lovasz`，backward 后执行一次 AdamW update。
2. 每 10 个 optimizer updates 打印 loss。
3. 每 500 个 updates 切到 eval，先跑完整 val，再跑完整 test。
4. 只有 val mIoU 超过历史最佳才保存纯模型 state dict；best 选择条件没有直接使用 test 指标。
5. 但是 test 被反复查看并写入 best-round 摘要，仍构成测试集窥视，不能用于严格复现。

因此正式训练不能直接执行该官方 Trainer。后续独立 trainer 会导入未修改的官方 Dataset、UNet、Lovasz 和 Evaluator，但只暴露 train/val 参数；test 评价继续使用已经存在的独立 `tools/evaluate_unet_testset.py`。

## 6. RTX 3050 Ti 4GB 资源适配方案

优先顺序：

### A. 论文/代码最接近方案（需要更大显存时转移 GPU）

- 14 事件，真实 batch 16，crop 640，FP32。
- AdamW、lr `1e-4`、weight decay `5e-3`。
- 代码损失 `CE + 0.75 Lovasz`。
- 50000 optimizer updates；每 500 步只跑 val。
- 这是最接近当前官方可执行基线的方案，但能否在 4GB 上运行尚未证实，预计有较高 OOM 风险。

### B. 4GB 资源适配方案（不等同论文）

- 数据范围仍为全部 2918 个训练 ID，不减少事件或样本范围。
- crop 继续 640，不用缩小输入掩盖显存问题。
- micro-batch 1、gradient accumulation 16，使每个 optimizer update 累积 16 个样本。
- optimizer update 仍为 50000，样本曝光仍为 800000。
- 烟测先用 AMP 确认链路，再做 FP32 micro-batch 1 探测；正式训练优先 FP32，只有
  FP32 OOM 或余量不足时才采用并明确标注 AMP。
- validation batch 1，workers 0；每 500 optimizer updates 验证一次。
- 只按 val mIoU 选 best，训练阶段没有 test 参数。

必须单独披露的差异：

- micro-batch 1 会让 UNet 的 BatchNorm 每次只看到一个样本；gradient accumulation 无法恢复真实 batch 16 的 BN 统计。
- AMP 改变数值精度。
- seed=1337 和 deterministic cuDNN 是新增设置，论文没有报告原始 seed。
- workers 0 会改变数据增强的具体随机流和吞吐。
- 完整 checkpoint/原子保存是工程增强，而官方只保存 model state。

所以方案 B 的正确名称是“ESSD standard_ML 14 事件资源适配复现”，不得称为完全一致复现。

## 7. 真实数据训练烟测

烟测目标仅包括：官方 loader 能读取真实 TIFF、UNet 前向输出 `N×4×640×640`、CE 和 Lovasz 为有限值、backward 成功、AdamW update 后参数发生变化、CUDA 峰值被记录。

Windows PowerShell：

```powershell
Set-Location 'D:\BRIGHT-Reproduction\BRIGHT'
git switch repro/essd-unet-baseline
git pull origin repro/essd-unet-baseline

python .\tools\smoke_train_unet_standard_ml.py `
  --dataset-root 'D:\BRIGHT-Reproduction\data' `
  --device cuda `
  --steps 2 `
  --sample-limit 8 `
  --micro-batch-size 1 `
  --crop-size 640 `
  --seed 1337 `
  --amp `
  --report '.\outputs\standard_ml_train_smoke\training_smoke_amp.json'
```

建议先用 AMP 确认训练链路，再去掉 `--amp` 做一次 FP32 显存探测。两次报告放在不同文件名，避免覆盖，例如 `training_smoke_amp.json` 和 `training_smoke_fp32.json`。

成功标准：

- `status` 为 `passed`；
- train/val 数量为 2918/450，均含 14 个事件且交集为 0；
- `test_list_accessed` 为 `false`、validation/test TIFF 读取数为 0；
- 每步 input/output 分别为 `1×6×640×640` 和 `1×4×640×640`；
- CE、Lovasz 和 total loss 都为有限值；
- `parameter_update.changed` 为 true；
- 记录峰值 allocated/reserved CUDA bytes、设备和耗时。

烟测每个 micro-batch 都立即 update，不模拟 accumulation 16；它只能证明训练链路可用，loss 和速度不是正式实验结果。

## 8. 烟测通过后的正式 trainer 验收条件

收到人工确认后才实现/启动正式 trainer。最低要求：

1. 固定读取官方 train/val；命令行和配置中不存在 test-list 参数。
2. 显式终止于 50000 optimizer updates，并记录 micro steps、samples seen 和有效 batch。
3. 保持模型、标签、crop 640、官方增强、AdamW、lr、weight decay 和代码 loss 不变。
4. 每 500 optimizer updates 验证；只按 val mIoU 保存 best。
5. `latest.pt` 保存 model、optimizer、AMP scaler、update、micro-step、Python/NumPy/torch/CUDA RNG、数据顺序状态、配置、git commit 及 train/val SHA-256。
6. `best_val_miou.pt` 与 `latest.pt` 分开；原子写入，保留少量周期快照。
7. 正式训练完成且 best 固定后，才首次读取 `test_set.txt`，用官方整幅 1024×1024 评价一次。

## 9. 正式测试阶段（现在不要执行）

最终评价必须：

- 使用官方 878 ID `test_set.txt`，不加 `--paper-nine-only`；
- 使用冻结后的本次训练 best checkpoint，而不是官方 standard_ML checkpoint；
- 整幅 1024×1024、batch 1、无 crop、无滑窗、无 resize、无 TTA；
- 使用官方 Evaluator 逻辑输出 F1loc、F1clf、OA、mIoU、各类 F1/IoU 和混淆矩阵；
- 记录 checkpoint MD5、split SHA-256、失败样本和样本数；
- 只有此结果才与 ESSD Table 5 做逐项差值。

已有的 `outputs/official_unet_paper9_test`、9 事件预测图、报告和派生列表不得删除或覆盖。14 事件主实验统一写入 `outputs/standard_ml_*`，避免目录混用。

## 10. 环境、耗时、显存和磁盘记录

沿用已经验证的 Windows 环境：PyTorch `2.5.1+cu121`、RTX 3050 Ti Laptop 4GB。不要为了匹配仓库 `torch==2.2.1` 而破坏当前可用 CUDA 环境；其余依赖使用 `requirements-reproduction.txt`。

训练时间不能从 0.353 秒推理时间推断。烟测后用 CUDA synchronize 测量 warm-up 后至少 20–50 个 optimizer updates，再单独测完整 450 样本 validation；若每 500 步验证，完整训练共有 100 次 validation。正式估算需包含训练、validation、checkpoint I/O 和 Windows 显卡共享占用。

官方纯模型权重为 124274886 bytes（约 118.5 MiB）。完整 AdamW checkpoint 还包含模型和两份 moment，通常至少约纯权重的 3 倍，最终以第一次实测文件为准。建议只保留 best、latest 和最近两个周期快照；正式启动前用 `Get-PSDrive D` 再确认空间。

每个正式 run 至少保存：

```text
config.json
git_commit.txt
train_val_split_hashes.json
environment-pip-freeze.txt
environment-nvidia-smi.txt
train.jsonl
validation.jsonl
best_val_miou.pt
latest.pt
final_test/evaluation_report.json
```

## 11. 当前尚未解决的问题

- 论文公式 `CE + Lovasz` 与当前源码 `CE + 0.75 Lovasz` 无法从公开材料进一步消解。
- 论文没有明确 crop 数值、增强概率、验证频率、seed、初始化和 checkpoint 内部格式；这些细节来自代码/README或仍未知。
- 官方发布 checkpoint 不含训练 ID、split 哈希、seed 或源码 commit，不能作为本次独立训练的可验证 provenance。
- 4GB 上 FP32/AMP micro-batch 1 的真实训练峰值尚未测量。
- 尚未决定烟测后使用更大 GPU 的真实 batch 16，还是接受带 BatchNorm/AMP 差异的资源适配方案。
- 独立正式 trainer 尚未实现；必须在烟测结果和资源路线确认后再实现，当前不能启动完整训练。
