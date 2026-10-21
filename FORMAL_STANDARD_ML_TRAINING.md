# BRIGHT ESSD 2025 standard_ML UNet 正式训练

## 目标与实现

本脚本按官方 `standard_ML` 训练列表运行 14 事件 UNet。清单数量固定校验为 train 2918、val 450；训练清单扩展至 800,000 次样本曝光，对应 batch 16 时 50,000 次 optimizer update。验证每 500 次成功更新进行一次，只按官方 `Evaluator` 的 mIoU 选择 best。训练脚本没有 test-list 参数，也不会读取 test manifest。

固定保留官方 `UNet(6,4)`、AdamW、learning rate `1e-4`、weight decay `0.005`、crop 640、官方增强，以及源码实际的 `CE + 0.75 × lovasz_softmax`。不使用 scheduler。

运行工具：`tools/train_unet_standard_ml.py`。官方模型、Dataset、增强、Lovasz、Evaluator 和原始划分均未修改。

## 4GB 显存适配与论文差异

默认配置使用 micro-batch 1、gradient accumulation 16、AMP。每 16 个 micro-batch 执行一次 AdamW 更新；如果所有更新成功，800,000 个样本曝光对应 50,000 次更新。GradScaler 按标准流程 unscale、step、update；一次溢出会跳过该次参数更新、降低 scale、清空梯度并记录后继续。连续 8 次溢出则保存 `latest.pt` 并安全停止。CE、Lovasz 或总 loss 非有限、FP32 梯度非有限、OOM 和其他错误均停止并写 `failure_report.json`。

该设置属于资源适配训练，不能称为完全复现论文原始 batch 16：

- AMP 的数值路径不同于官方 FP32。
- BatchNorm 在每个 micro-batch 1 上更新统计；累积梯度不能恢复真实 batch 16 的 BatchNorm 统计。
- 官方 Lovasz 实现以当前 batch 展平像素。micro-batch 1 下分别计算 16 个单样本 Lovasz 并平均梯度，不等于在一个 batch 16 上统一计算 Lovasz。函数与权重系数保留，但批内聚合口径改变。
- 用户指定的 seed 1337 用于确定性续训；论文未报告该 seed。
- AMP 溢出会消耗这部分样本曝光预算但跳过 optimizer update。因此训练以 800,000 次曝光为停止条件，溢出时成功更新数可能略少于 50,000；报告会同时记录曝光数、尝试次数、成功更新和跳过数。

## Windows 环境

当前已验证环境为 Windows 11、Python 3.12.2、PyTorch `2.5.1+cu121`、RTX 3050 Ti Laptop 4GB。建议继续使用已通过真实数据烟测的 CUDA 环境，不为满足旧版 requirements 中的 torch pin 重装 CPU 版。其他依赖见 `requirements-reproduction.txt` 和 `README_REPRODUCTION.md`。

## 更新代码

```powershell
Set-Location 'D:\BRIGHT-Reproduction\BRIGHT'
git switch repro/essd-unet-baseline
git pull --ff-only origin repro/essd-unet-baseline
git status --short --branch
Get-PSDrive D
```

确认数据路径仍为 `D:\BRIGHT-Reproduction\data`，并为正式 run 使用全新输出目录。脚本会检查官方 train/val 清单 hash、样本数量、14 个事件、交集，以及对应 train/val TIFF 三元组是否存在。

## 正式训练启动

```powershell
python .\tools\train_unet_standard_ml.py `
  --dataset-root 'D:\BRIGHT-Reproduction\data' `
  --output-dir '.\outputs\standard_ml_unet_formal' `
  --device cuda `
  --sample-exposures 800000 `
  --micro-batch-size 1 `
  --gradient-accumulation-steps 16 `
  --crop-size 640 `
  --learning-rate 0.0001 `
  --weight-decay 0.005 `
  --validation-every-updates 500 `
  --checkpoint-every-updates 500 `
  --max-consecutive-amp-skips 8 `
  --seed 1337 `
  --amp
```

建议使用 PowerShell 窗口持续记录日志：

```powershell
python .\tools\train_unet_standard_ml.py `
  --dataset-root 'D:\BRIGHT-Reproduction\data' `
  --output-dir '.\outputs\standard_ml_unet_formal' `
  --device cuda --sample-exposures 800000 `
  --micro-batch-size 1 --gradient-accumulation-steps 16 `
  --crop-size 640 --learning-rate 0.0001 --weight-decay 0.005 `
  --validation-every-updates 500 --checkpoint-every-updates 500 `
  --max-consecutive-amp-skips 8 --seed 1337 --amp *>&1 |
  Tee-Object -FilePath '.\outputs\standard_ml_unet_formal\console.log'
```

输出文件：

```text
outputs/standard_ml_unet_formal/
  config.json
  split_hashes.json
  train.jsonl
  validation.jsonl
  latest.pt                 # model + AdamW + GradScaler + RNG + sample cursor
  best_model.pth            # raw official UNet state_dict, selected by val mIoU
  best_training_state.pt    # resumable state at best validation mIoU
  training_report.json
  failure_report*.json     # only if execution stops with an error; older files are kept
```

`latest.pt` 每 500 次 update attempt 原子更新；checkpoint 只在 16 个 micro-step 的 accumulation 边界保存。磁盘上仅保留一个 `latest.pt` 和一个 best resumable checkpoint，另有较小的纯模型 `best_model.pth`。验证日志每 500 次成功 optimizer update 写入。训练结束时运行最后一次 val（若最近 update 尚未验证），不读取 test。

## 断点续训

普通中断后，检查 failure report 和最后 checkpoint，再用相同配置与输出目录续跑：

```powershell
python .\tools\train_unet_standard_ml.py `
  --dataset-root 'D:\BRIGHT-Reproduction\data' `
  --output-dir '.\outputs\standard_ml_unet_formal' `
  --resume '.\outputs\standard_ml_unet_formal\latest.pt' `
  --device cuda --sample-exposures 800000 `
  --micro-batch-size 1 --gradient-accumulation-steps 16 `
  --crop-size 640 --learning-rate 0.0001 --weight-decay 0.005 `
  --validation-every-updates 500 --checkpoint-every-updates 500 `
  --max-consecutive-amp-skips 8 --seed 1337 --amp
```

训练配置和 train/val manifest hash 必须与 checkpoint 一致。一般进程中断会丢失最近一个 checkpoint 后尚未保存的工作量，最多约 499 次 update attempt。若因连续 AMP 溢出或非有限 loss 停止，先查看 `failure_report.json` 和末尾日志；不要仅为继续而提高阈值或忽略异常。坏 loss 尚未完成的 accumulation 不保存，续训会从最近完整 checkpoint 位置重新读取。

## 训练结束后的测试集评价

仅在 `training_report.json` 显示 `completed_sample_exposure_budget`、best checkpoint 存在且训练配置已冻结后执行一次。该命令使用官方 878 ID test list 和既有独立评价脚本：

```powershell
python .\tools\evaluate_unet_testset.py `
  --dataset-root 'D:\BRIGHT-Reproduction\data' `
  --checkpoint '.\outputs\standard_ml_unet_formal\best_model.pth' `
  --test-list '.\bda_benchmark\dataset\splitname\standard_ML\test_set.txt' `
  --device cuda `
  --output-dir '.\outputs\standard_ml_unet_formal\test_evaluation'
```

测试运行独立于训练命令。训练阶段不得调用该命令或读取 test 清单。

## 时间与风险估计

提供的 1000 步失败日志尾部中，update 159–188 的 30 步 `cycle_seconds` 中位数约为 0.347 秒，包含数据读取与前向/反向；其中 compute 中位数约 0.243 秒。按 800,000 个 micro-step 直接外推，训练计算约 277,638 秒，即 77.1 小时（约 3.2 天）。这是短日志尾部的初始估计，不包含约 100 轮完整验证、checkpoint 写盘、系统负载、缓存波动和溢出跳过影响；实际需要按天预留，验证额外时长未实测。

最近日志尾部的单步峰值约 1.70 GB allocated、1.94 GB reserved，说明当前配置低于 4GB，但 Windows 桌面和其他进程会共享显存。全幅验证使用 batch 1、1024×1024，之前单样本整图推理约 1 GB 峰值；仍应关闭占用 GPU 的应用。若出现 OOM，failure report 会记录显存与 checkpoint 状态。必要时可在确认配置变化后将 AMP 关掉做资源路线调整，但 FP32 速度/显存和数值轨迹不同；micro-batch、accumulation、精度变化都必须在最终报告披露。

