# BRIGHT 1000 步试训练：AMP/FP32 有限性诊断

## 当前证据与定位

本工作区没有 `outputs/standard_ml_pilot_1000/` 原始目录。收到的粘贴文本包含 failure report 和 `training_log.jsonl -Tail 30`。因此，下面对原运行的判断只基于这段报告/日志尾部，不声称读过完整 1000 步日志。

可确认的事实：

- 最后一条成功落盘的日志是 update 188，样本 `turkey-earthquake_00000194`。它的 CE=`0.6217956543`、Lovasz=`0.4813073874`、总 loss=`0.9827761650`，均为有限数。
- 异常发生在 update 188 之后的下一次尝试，按脚本从 1 开始计数应是 update 189；失败尝试的样本 ID 没有写进日志。
- 报错为 `Non-finite gradient in parameter final_conv.weight`。旧脚本执行顺序是 `scaler.scale(loss).backward()` → `scaler.unscale_(optimizer)` → 手工检查各参数梯度 → `scaler.step()` → `scaler.update()`。报错发生在手工检查阶段，因此 `scaler.step()` 和 `scaler.update()` 都没有执行，异常 update 没有更新模型参数。
- 到达梯度检查意味着前向 logits 检查通过，而且 `finite_float(total_loss)` 已通过。因此脚本没有检测到非有限输出或总 loss。旧日志没有异常 update 的 CE/Lovasz 分项、Scaler scale、梯度中 NaN/正负 Inf 的数量，也没有失败样本 ID。
- CE 与 Lovasz 在该实现中均为非负损失；总 loss 有限与二者正权重相加，强烈支持分项也是有限值，但旧脚本没有在失败前单独记录/检查分项，故报告不把它写成直接观测值。
- failure report 记载 CUDA peak allocated=`1,699,380,224` bytes、peak reserved=`1,941,962,752` bytes。失败类型是 `FloatingPointError`，不是 OOM。

这说明非有限值出现在反向传播产生的梯度中，并在 GradScaler unscale 后被检查出来。GradScaler 的 unscale 阶段会登记 Inf/NaN；旧脚本随后立刻抛错，没到 scaler 的跳过更新和缩小 scale 流程。证据还不能区分 AMP scaled-gradient overflow、Lovasz/softmax 反向数值问题、权重状态累积、具体 crop 的标签分布或其他梯度不稳定原因。`final_conv.weight` 是第一个被检查到异常的参数，不等于根因一定在最后一层。

## 代码改动

`tools/train_unet_standard_ml_short.py` 现在在失败报告中附带当前尝试的 update、样本 ID/事件、失败阶段、logits 范围、CE/Lovasz/总 loss 与有限性、GradScaler scale、梯度 NaN/正负 Inf 数量和有限梯度绝对值上限。成功日志也记录 CE/Lovasz 有限性和 scaler 更新前后 scale。失败仍立即中止；没有跳过坏梯度，也没有关闭有限性检查。

新增的 `tools/diagnose_unet_amp_fp32.py` 会从相同 seed、相同默认 UNet 初始化和相同 official standard_ML 训练 ID 顺序分别重放 AMP 与 FP32。两支各自读取官方 Dataset 生成增强 crop；会记录增强后输入/标签 SHA-256，证明两支对照到同一批张量。它逐次记录样本 ID、logits 有限性/范围、CE/Lovasz/总 loss、梯度异常类别与数量、Scaler scale 和失败阶段。发现坏梯度时不执行 optimizer step，并立即停止该分支。AMP 异常时只调用一次 scaler update 来观察 scale 是否下降，之后销毁该分支，不会继续训练。脚本没有 val/test 参数，也不读取相应清单。

这个对照从 seed 初始化重新回放，而不是从第 188 步权重恢复。只有当原运行使用相同 seed、源码、PyTorch/CUDA 与确定性设置时，才有机会重现相同轨迹；AMP 与 FP32 在有更新后参数会自然分叉。因此该测试能比较同一数据序列下两种精度的有限性，不能保证复刻原失败权重状态。

## Windows PowerShell 命令

先同步分支，给诊断结果使用全新目录：

```powershell
Set-Location 'D:\BRIGHT-Reproduction\BRIGHT'
git switch repro/essd-unet-baseline
git pull --ff-only origin repro/essd-unet-baseline

python .\tools\diagnose_unet_amp_fp32.py `
  --dataset-root 'D:\BRIGHT-Reproduction\data' `
  --output-dir '.\outputs\standard_ml_amp_fp32_diagnostic_250' `
  --device cuda `
  --max-updates 250 `
  --micro-batch-size 1 `
  --crop-size 640 `
  --learning-rate 0.0001 `
  --weight-decay 0.005 `
  --seed 1337

if ($LASTEXITCODE -ne 0) {
  throw 'AMP/FP32 diagnostic process failed; inspect its console error.'
}

$Report = Get-Content '.\outputs\standard_ml_amp_fp32_diagnostic_250\amp_fp32_diagnostic_report.json' -Raw | ConvertFrom-Json
$Report.branches.amp.failure
$Report.branches.fp32.failure
$Report.runtime.paired_batches_with_identical_augmented_tensor_hashes
```

脚本每支最多 250 次更新；出现非有限 loss/gradient 即停。若 AMP 遇到 CUDA OOM，会记录该支失败并跳过 FP32 分支，避免继续占用已不足的显存。这个命令是诊断回放，不是正式训练，也不访问 test。

## 如何读结果

| 观察到的结果 | 结论与下一步 |
|---|---|
| AMP 在近似 update 189 出现 Inf/NaN，FP32 对同一批输入通过 | AMP 数值范围是主要嫌疑之一。检查 AMP failure 中 Inf/NaN 数量、scale 和 `grad_scaler_scale_decreased`；若 scale 下降，说明 GradScaler 识别到了溢出。下一步先保存这份诊断证据，再设计仅针对数值精度的受控验证；不凭该结果直接改学习率或跳过更新。 |
| AMP 与 FP32 在同一 update、相同输入 hash 上均出现非有限梯度 | 问题不只由 AMP 引入。核对报告中的 loss 分项、标签值和异常梯度参数；再围绕该 train sample 检查 TIFF、增强 crop 和官方 Lovasz 计算，不修改官方文件。 |
| AMP 与 FP32 的输入 hash 不同 | 对照无效；先查 seed、源码版本、PyTorch/CUDA 或数据集增强随机流差异，再比较数值结果。 |
| 两支到 250 步都通过 | 只能说明从当前初始化、seed 和前 250 个样本 crop 没能复现。检查与原运行的 seed/环境是否一致；若一致，再用同样受限的更长诊断上限扩展复现区间，不恢复失败训练继续跑。 |
| FP32 失败、AMP 通过 | 检查两支首次失败 update 的 loss、样本与梯度报告；两条参数轨迹在此前更新后已不同，不能把后续不同样本行为解释成精度的单变量因果。 |

在定位出相同失败样本/批次之前，不恢复 pilot checkpoint 长时间训练，也不调整学习率。诊断报告只用于排查数值有限性，不报告模型精度。
