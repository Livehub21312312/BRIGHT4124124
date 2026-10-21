# BRIGHT standard_ML 14 事件短程试训练

> 本文档只描述约 50 次 optimizer update 的诊断性试训练。它不是完整训练、不是论文结果，也不会读取 validation 或 test 列表/影像。正式主线仍是 ESSD 2025 的 14 事件 `standard_ML` 划分；已有 9 事件实验保持独立且不删除。

## 1. 配置核对结论

依据 ESSD 2025 正式论文、仓库根 README、`bda_benchmark/sh_file/standard_ML_split/train_UNet.sh`、`train_UNet.py`、Dataset 和增强实现：

| 项目 | ESSD 2025 正文 | 官方可执行源码/命令 | 本次 50-update 试训练 |
|---|---|---|---|
| 数据范围 | 14 个事件的 standard ML 实验 | train/val/test = 2918/450/878 | 只允许读取 2918 个 train ID；不打开 val/test list |
| optimizer | AdamW | AdamW | AdamW，不变 |
| learning rate | `1e-4` | 恒定 `1e-4` | `1e-4`，不变 |
| weight decay | `5e-3` | `5e-3` | `5e-3`，不变 |
| scheduler | 未报告 | 没有 scheduler | 没有 scheduler |
| batch size | 16 | shell/README 为 16；Python 参数默认 8，但正式命令覆盖为 16 | micro-batch 1，且不做梯度累积；与论文不同 |
| crop | 论文只明确 random cropping | shell/README 为 640 | 640，不变于官方命令 |
| 训练步数 | 50,000 iterations | `max_iters=800000` 是 Dataset 展开后的样本曝光数；batch 16 正好产生 50,000 updates | 默认 50 updates、50 次样本曝光，仅诊断 |
| loss | 论文公式为 CE + Lovasz | `CE + 0.75 × Lovasz` | 保留可执行源码的 `0.75`；论文/代码差异继续披露 |
| 增强 | random crop、flip、90° rotation | random crop；水平/垂直翻转各约 0.5；rotation 每次在 90/180/270° 中选一个，不含 0° | 直接复用官方 Dataset/增强，不修改 |
| AMP | 未报告 | 没有 autocast/GradScaler，即 FP32 | 默认开启，是 4GB 显存适配差异 |
| 梯度累积 | 未报告 | 没有，每个 batch 立即更新 | 没有，每个 micro-batch 立即更新 |
| 初始化 | 未报告 | 新建 `UNet(6, 4)`，使用 PyTorch 层默认初始化；未设置 seed | 同一模型和默认层初始化算法，但新增固定 seed 1337 |
| 验证/保存 | 正文未给出频率 | 每 500 updates 跑 val 和 test；按 val mIoU 保存纯模型权重 | 不验证；每 25 updates 保存完整恢复状态，结束后做恢复自检 |

`max_iters` 的换算必须保持清楚：

```text
optimizer_updates = ceil(sample_exposures / batch_size)  # drop_last=False
800000 / 16 = 50000
```

因此不能把官方 `max_iters=800000` 当作 80 万次 optimizer update。反过来，将 batch 改为 1 后仍运行 80 万次即时更新也不等价于论文训练。

官方 `train_UNet.py` 每 500 步读取 test 并报告 test 指标，虽然 best checkpoint 条件只看 val mIoU，仍会造成测试集窥视。本次独立脚本没有 val/test 路径参数，只导入以下未修改实现：

- `bda_benchmark/dataset/make_data_loader.py`
- `bda_benchmark/dataset/imutils.py`
- `bda_benchmark/model/UNet.py`
- `bda_benchmark/util_func/lovasz_loss.py`

## 2. 已完成烟测与本次试训练的区别

Windows 本地已经完成 2-update 真实数据烟测：PyTorch `2.5.1+cu121`、RTX 3050 Ti Laptop 4GB、AMP、micro-batch 1、crop 640；两次更新成功，参数变化验证通过，峰值 allocated/reserved 分别为 1,700,690,944 / 1,941,962,752 bytes，单步观测约 0.752 秒，且未访问 test list。

两步数据只能证明训练链路可用，不能视为稳定吞吐。新增的 `tools/train_unet_standard_ml_short.py` 默认执行 50 次即时更新，并额外完成：

- 每步记录 CE、Lovasz、总 loss、样本 ID、输入/输出形状、数据读取耗时、计算耗时和 CUDA 峰值；
- 任一输出、loss 或梯度出现 NaN/Inf 时立即停止；CUDA OOM 也立即停止；
- 比较运行前后同一模型参数的 SHA-256 和最大绝对变化；
- 第 25、50 步原子保存 checkpoint；
- checkpoint 包含 model、AdamW optimizer、AMP scaler、update、样本位置、固定样本顺序和 Python/NumPy/PyTorch/CUDA RNG 状态；
- 结束后在 CPU 上创建全新 UNet/optimizer/scaler，以 `strict=True` 恢复并逐项比较状态；失败则整次试训练标记失败。

脚本使用固定乱序和 seed 是可重复性工程增强。它不改变官方模型、loss、标签或增强代码，但其随机序列并不等同于未记录 seed 的论文运行。

## 3. Windows PowerShell 执行

先同步分支并确认磁盘：

```powershell
Set-Location 'D:\BRIGHT-Reproduction\BRIGHT'
git switch repro/essd-unet-baseline
git pull --ff-only origin repro/essd-unet-baseline
Get-PSDrive D
```

使用一个新输出目录，避免覆盖既有烟测和 9 事件结果：

```powershell
$RunDir = '.\outputs\standard_ml_short_train_50'

python .\tools\train_unet_standard_ml_short.py `
  --dataset-root 'D:\BRIGHT-Reproduction\data' `
  --output-dir $RunDir `
  --device cuda `
  --optimizer-updates 50 `
  --micro-batch-size 1 `
  --crop-size 640 `
  --learning-rate 0.0001 `
  --weight-decay 0.005 `
  --warmup-updates 5 `
  --checkpoint-every 25 `
  --num-workers 0 `
  --seed 1337 `
  --amp

if ($LASTEXITCODE -ne 0) {
  throw "Short training failed; inspect $RunDir\failure_report.json"
}

Get-Content "$RunDir\short_training_report.json"
```

默认输出：

```text
outputs/standard_ml_short_train_50/
  training_log.jsonl
  checkpoint_step_000025.pt
  checkpoint_step_000050.pt
  short_training_report.json
```

若 OOM、非有限值或恢复验证失败，进程返回非零，并生成 `failure_report.json`。不要在失败后自动降低 crop、改变 loss 或跳过样本；先保留报告再决定下一项受控改动。

`short_training_report.json` 至少应满足：

- `status == "passed"`；
- `paper_result == false`；
- train count 为 2918，含 14 个事件；
- `validation_list_accessed` 和 `test_list_accessed` 都是 `false`；
- 完成 50 updates，输入/输出为 `1×6×640×640` / `1×4×640×640`；
- 参数 `changed == true`；
- checkpoint 中 model、optimizer、AMP scaler 三项恢复检查均为 true。

脚本还支持真正的中断续跑。`--optimizer-updates` 始终表示最终总目标，例如从第 25 步文件恢复到总计 50 步：

```powershell
python .\tools\train_unet_standard_ml_short.py `
  --dataset-root 'D:\BRIGHT-Reproduction\data' `
  --output-dir '.\outputs\standard_ml_short_train_resume_demo' `
  --device cuda --optimizer-updates 25 --amp

python .\tools\train_unet_standard_ml_short.py `
  --dataset-root 'D:\BRIGHT-Reproduction\data' `
  --output-dir '.\outputs\standard_ml_short_train_resume_demo' `
  --device cuda --optimizer-updates 50 --amp `
  --resume '.\outputs\standard_ml_short_train_resume_demo\checkpoint_step_000025.pt'
```

不必为了本次任务额外执行续跑演示：默认 50 步结束时已经自动验证三类状态可恢复。上面的命令只用于需要验证跨进程继续训练时。

## 4. 时长估算与使用方法

现有 0.752 秒是仅两步得到的观察值，不能当稳定速度。如果只做算术外推、并明确忽略启动、数据加载波动、checkpoint 和其他开销：

| 当前 micro-batch 1 配置 | 暂定算术值 |
|---|---:|
| 50 updates | 37.6 秒 |
| 1000 updates | 752 秒，即 12 分 32 秒 |

这两项不是运行承诺。50 步脚本会排除前 5 个 warm-up updates，以余下步骤的 `data + compute` 中位数重新给出 50/1000 步外推；应以这份实测报告为下一阶段估算依据。短程脚本的 compute 计时包含输出/loss/梯度有限值检查，因此它是带诊断保护的保守观测，不是未经插桩的官方 trainer 纯吞吐；总 wall time 还单独包含初始化、样本检查、checkpoint 和恢复自检。

论文/官方 batch 16 的约 50,000 updates **不能**用 `50000 × 0.752` 估算：0.752 秒对应 micro-batch 1，batch 16 的吞吐、显存行为和 BatchNorm 都不同，而且 4GB GPU 很可能无法容纳真实 batch 16。正确表达是：

```text
T_original = 50000 × stable_seconds_per_true_batch16_update
             + 100 × full_validation_time
             + checkpoint/startup/OS overhead
```

需要在能运行真实 batch 16 的目标 GPU 上先测稳定步时间才能代入。若以后接受 micro-batch 1 + accumulation 16 的资源适配方案，仅按现有两步值粗算 80 万个 micro steps 就是约 601,600 秒（167.1 小时、约 7 天）的计算量级，仍未计 100 次 validation、数据读取和保存；这不是原始 batch 16 的时间估算，也不是正式排期。

额外开销来源包括 TIFF 解码和 Windows 文件缓存、首次 CUDA kernel/cuDNN warm-up、AMP scaler、AdamW 状态首次创建、后台显存占用、每 500 步的 450 张整图 validation、checkpoint 写盘和杀毒软件扫描。短程结果只应用来校准，不能外推论文精度。

## 5. 当前边界与下一步门槛

- 本提交不运行 50 步，不生成任何新实验指标；需要在用户的 Windows 真实数据/GPU 上执行。
- 本次短程配置没有 gradient accumulation，因此 50 updates 只曝光 50 个样本，不等于论文 batch 16 的 800 个样本曝光。
- AMP、micro-batch 1、seed 1337、workers 0、完整 checkpoint 都是本地适配/工程增强。
- 50 步通过后，仍需人工选择：转到能运行真实 batch 16 的 GPU，或接受 micro-batch 1 + accumulation 16 的资源适配轨道。后者的 BatchNorm 统计不等价于真实 batch 16。
- 未经确认不启动正式训练，不访问官方 test 列表，也不以 test 指标调参或选 checkpoint。
