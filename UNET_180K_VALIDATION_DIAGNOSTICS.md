# BRIGHT 180k UNet 验证集诊断

该工具只对已保存的 `best_model.pth` 做推理，不训练、不反传、不更新权重；仅读取官方 `standard_ML/val_set.txt` 的 450 个验证样本，不提供 test-list 参数，也不访问测试样本。它会用官方 Dataset、UNet 和 Evaluator 逻辑产生总体与类别指标、标签/预测分布、混淆矩阵、逐样本类别统计和代表性四联图。

## Windows PowerShell

```powershell
Set-Location 'D:\BRIGHT-Reproduction\BRIGHT'
git switch repro/essd-unet-baseline
git pull --ff-only origin repro/essd-unet-baseline

python .\tools\diagnose_unet_validation.py `
  --dataset-root 'D:\BRIGHT-Reproduction\data' `
  --checkpoint '.\outputs\standard_ml_unet_180k_resource_adapted\best_model.pth' `
  --training-report '.\outputs\standard_ml_unet_180k_resource_adapted\training_report.json' `
  --val-list '.\bda_benchmark\dataset\splitname\standard_ML\val_set.txt' `
  --output-dir '.\outputs\standard_ml_unet_180k_diagnostics' `
  --device cuda
```

输出目录必须不存在或为空，脚本拒绝覆盖非空目录。成功后包含：

- `diagnostic_report.json`：best 验证 checkpoint、严格加载结果、官方总体/每类指标、类别分布、三类混淆矩阵、设备和运行范围。
- `per_sample_class_counts.csv`：450 个样本的事件和真实/预测四类像素数。
- `representative_samples\*_diagnostic.png`：灾前光学、灾后 SAR、真实标签、预测标签四联图；标签颜色为背景白、intact 绿、damaged 米黄、destroyed 红。

## 如何读诊断结果

训练报告中的 `training_state.validation_history` 可能包含多次验证。本工具按其中 `miou` 最大值找最佳记录，并与 `training_state.best_update` / `best_val_miou` 并列展示；不会把最后一次验证误写成 best。若报告不存在或没有历史记录，对应字段会明确显示不可用。若报告的 best update 与历史最大 mIoU 对不上，应先核查报告与 checkpoint 是否来自同一 run。

`best_model.pth` 预期为原始 UNet `state_dict`（键名到 tensor），本工具实例化官方 `UNet(6,4)` 并使用 `strict=True` 加载；格式、shape、键不匹配时会停止，不做宽松加载。报告保存 checkpoint 大小、SHA-256 和每个参数的 shape。

标签 ID 固定为 0 background、1 intact、2 damaged、3 destroyed。`ground_truth_distribution` 与预测分布按有效真实标签像素统计；预测类比例以同一有效像素数为分母。总四类 confusion matrix 的行为真实类、列为预测类，包含全部有效像素。另有官方 damage evaluator confusion matrix：只在真实建筑像素（官方 localization label > 0）上统计 1/2/3。

官方 `Damage_F1_score()` 分别计算 intact/damaged/destroyed 的 F1；官方 `F1clf` 再对这三个值取调和平均。因此，任一损伤类别 F1 为 0，就会令 `F1clf` 为 0。需要结合 per-class F1、真实/预测计数和 damage confusion matrix 判断是漏检（某真实损伤类别大多落到 intact/背景）、误报，还是该类别预测完全缺失。`mIoU` 和 `OA` 则来自全像素四类 confusion matrix。

## 当前已确认与待确认

从源码可以确认：Dataset 不重映射 1/2/3 分类标签；仅另造 localization label，把 2、3 映成建筑类 1。UNet 输出通道 0–3 直接 `argmax` 为预测标签。官方验证只在 `labels_loc > 0` 的像素上计算 damage F1，官方 evaluator 的 `Damage_F1_score()` 包含类别 1、2、3，且没有类别权重。CE + 0.75 × Lovász 使用原始分类标签训练；CE 的 `ignore_index` 默认不是 255，而 Lovász 显式忽略 255。

类别不均衡、background/intact 优势、稀少的 damaged/destroyed 样本、资源适配 batch/累积差异或 AMP 数值差异都可能影响损伤分类，但在读取这次 run 的真实 report、混淆矩阵和验证集预测前都只是解释假设。当前 Codex 工作区不挂载 Windows 的 `D:` 盘，也没有该 run 的 report/checkpoint/真实数据，因此不会虚构最佳 mIoU、checkpoint strict-load 结果、类别分布或诊断图片。请在 Windows 执行上述命令后，以生成的 JSON/CSV/PNG 作为本次 run 的证据。

**当前可给出的结论仅限源码事实**：若本地报告确实显示 `F1clf=0`，其必要数值原因是三个 damage 类 F1 中至少一个为零；究竟是哪一类、模型是否塌缩为 background/intact，必须由本工具的真实验证输出确认。当前最值得先做的事情是运行这次只读验证诊断，而不是改标签、修改官方 Evaluator 或依据测试集调参。诊断后再依据最弱类别的 FN/FP 和预测占比设计一项验证集可检验的改进。
