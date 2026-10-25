# BRIGHT standard_ML 类别加权微调 Pilot

此独立实验从现有 `standard_ml_unet_180k_resource_adapted/best_model.pth` 严格加载 UNet 参数，再以官方 train 清单及其标签做 20,000 次样本曝光的资源适配微调。它不是从头训练，也不是完整论文复现。训练/验证仍使用官方 14 事件 standard_ML 划分；测试清单不传入训练引擎。

## 类别权重和损失

只读取官方 2,918 个 train 标签，统计有效像素 0=background、1=intact、2=damaged、3=destroyed。对频率 `f_i` 计算 `1/sqrt(f_i)`，再用二分缩放系数 `s` 使 `sum(f_i * clip(s/sqrt(f_i), 0.25, 4.0)) = 1`。背景、原始逆平方根权重、归一化/截断后的权重、像素数和频率都会写入 `class_weights.json`。如果发现 255 标签，脚本会在训练前停止：当前基线交叉熵的官方调用没有设置 `ignore_index=255`，为避免悄悄改变其语义，不会擅自忽略这些像素。

训练损失是 `CrossEntropy(logits, labels, weight=class_weights) + 0.75 * official lovasz_softmax(softmax(logits), labels, ignore=255)`。权重只作用于 CE，Lovász 实现不改。AdamW 使用新建的 optimizer state；模型参数从基线最佳权重 `strict=True` 初始化。

验证曝光目标为 5k/10k/15k/20k。micro-batch=1、accumulation=16 时，验证只能在完整 optimizer update 边界运行，因此实际验证曝光为 5008、10000、15008、20000；JSONL 会同时记录请求节点和实际 `samples_seen`，不会混淆曝光数与更新数。预期 1,250 次 update attempts；AMP 溢出按本 pilot 的保护策略首次发生即记日志、保存可恢复 checkpoint 并停止，不静默跳过。

## Windows PowerShell 命令

```powershell
Set-Location 'D:\BRIGHT-Reproduction\BRIGHT'
git switch repro/essd-unet-baseline
git pull --ff-only origin repro/essd-unet-baseline

python .\tools\train_unet_class_weighted_pilot.py `
  --dataset-root 'D:\BRIGHT-Reproduction\data' `
  --baseline-dir '.\outputs\standard_ml_unet_180k_resource_adapted' `
  --output-dir '.\outputs\standard_ml_unet_class_weighted_pilot' `
  --baseline-diagnostics '.\outputs\standard_ml_unet_180k_diagnostics\diagnostic_report.json' `
  --device cuda `
  --seed 1337
```

若已有独立 `diagnostic_report.json` 不在上述路径，可以省略 `--baseline-diagnostics`；报告会使用任务中提供的基线数值，并注明来源。输出目录已存在且非空时，首次运行会拒绝覆盖。请不要删除旧目录来绕过保护；为新 run 选一个全新的 `--output-dir`。

## 输出文件

- `config.json`：预算、数据切分 hash、初始 checkpoint hash、损失及全部训练参数。
- `class_weights.json`：训练标签频率、像素数、原始及最终权重公式和向量、基线比较来源。
- `training_log.jsonl`：每个 micro-step 的损失、样本 ID、GradScaler、耗时和 update 记录。
- `validation_report.jsonl`：每个节点的 F1loc、F1clf、OA、mIoU、损伤类 precision/recall/F1、四类 IoU、像素分布及混淆矩阵。
- `best_model.pth`：只按验证 mIoU 选择的 raw UNet state dict。
- `best_training_state.pt`：最佳验证点的模型、AdamW、GradScaler、RNG 与进度。
- `latest.pt`：周期及最终保存的可恢复状态。
- `final.pt`：预算完成点的完整最终状态。
- `experiment_report.json`：预算、成功更新数、最佳/末次验证记录和相对基线差值。
- `failure_report.json`（若已存在则编号追加）：非有限 loss、AMP 梯度溢出、OOM 等停止原因。

比较重点是 damaged/destroyed 是否获得非零 recall/F1、F1clf 是否改善以及 mIoU 是否提高；OA 只作辅助指标。此 pilot 最多只能证明短程类别加权微调的趋势，不能替代完整训练或论文结果。

## 断点续训

确认 `failure_report*.json` 和 `latest.pt` 后，使用完全相同参数，并增加 `--resume`：

```powershell
python .\tools\train_unet_class_weighted_pilot.py `
  --dataset-root 'D:\BRIGHT-Reproduction\data' `
  --baseline-dir '.\outputs\standard_ml_unet_180k_resource_adapted' `
  --output-dir '.\outputs\standard_ml_unet_class_weighted_pilot' `
  --baseline-diagnostics '.\outputs\standard_ml_unet_180k_diagnostics\diagnostic_report.json' `
  --device cuda `
  --seed 1337 `
  --resume '.\outputs\standard_ml_unet_class_weighted_pilot\latest.pt'
```

恢复会核验类别权重、训练/验证清单 hash、配置及初始 checkpoint hash，并还原 optimizer、GradScaler、RNG 和曝光游标。不要更改数据、权重或其他参数后沿用旧 checkpoint。

## 常见停止情况

- CUDA 不可用：确认 Windows 上已安装与现有 CUDA 环境兼容的 PyTorch，再运行 `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"`。
- 缺 TIFF 或名单不符：确认数据根目录层级含 `pre-event`、`post-event`、`target`，并且 checkout 使用官方 standard_ML split 文件。
- strict load 错误：检查 baseline `best_model.pth` 是否是原始 UNet state dict、来自本项目且未被替换；脚本不会以 `strict=False` 降级加载。
- 255 标签：当前代码会在训练前明确停止。不要自行改标签；先检查数据标签来源和官方 CE 对 ignore label 的预期，再决定是否另立实验。
- AMP 溢出/OOM/非有限 loss：查看 `failure_report.json` 和最后的 `training_log.jsonl`。此 pilot 首次检测到 GradScaler 溢出即停止；OOM 不会自动缩小输入或 batch，避免静默改变配置。
- 输出目录非空：保留原实验，使用新的输出目录运行，不要覆盖已有文件。
