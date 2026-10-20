# BRIGHT 真实数据单样本烟测

本阶段只验证一组真实 TIFF 能否经官方数据加载器进入官方 UNet，并完成一次只读前向
传播。脚本不会遍历完整数据集，不会反向传播、更新参数、开始训练或修改任何 TIFF、
标签和划分文件。

> 当前 Codex 云环境无法访问 Windows 的 `D:\BRIGHT-Reproduction\data`。因此本文只
> 记录脚本及可在本地执行的命令，不把云端的代码级测试描述成真实数据运行结果。

## 1. 官方数据读取链路

原始实现位于 `bda_benchmark/dataset/make_data_loader.py`：

1. 按训练列表中的样本 ID 拼出三个固定文件名：
   `<id>_pre_disaster.tif`、`<id>_post_disaster.tif` 和
   `<id>_building_damage.tif`。
2. 灾前光学影像按 `imageio.imread(...).astype(float32)` 读取，只保留前 3 个波段。
3. 灾后 SAR 必须是二维单波段数组，然后用 `np.stack` 原样复制成 3 个通道。
4. 光学和复制后的 SAR 分别使用均值 `[123.675, 116.28, 103.53]`、标准差
   `[58.395, 57.12, 57.375]` 逐通道归一化，并转换为 `C×H×W`。因此 SAR 在
   归一化前的三个源通道相同，但归一化后的数值不同。
5. 训练入口把二者拼成 `[B,6,H,W]`；`bda_benchmark/model/UNet.py` 中的官方 UNet
   输出 `[B,4,H,W]`。
6. 分类标签必须是二维的 `0/1/2/3`；定位标签是其副本，并把 `2/3` 改成 `1`。

`tools/smoke_test_real_data.py` 直接导入以上两个官方类，不复制或修改它们。为了既检查
原始文件又避免随机增强，脚本先独立读取完整 TIFF 并核查尺寸、波段、dtype、有限值和
标签取值，随后以官方 Dataset 的 `type="test"` 路径读取同一训练样本。默认在完整
1024×1024 数据检查通过后，对归一化后的真实数据做确定性 640×640 中心裁剪，只执行
一次 `model.eval()` + `torch.inference_mode()` 前向传播。

自动选样本时，脚本按官方 `standard_ML/train_set.txt` 的顺序查找第一个三文件均存在
且属于以下论文 9 事件的样本：

`bata-explosion`、`beirut-explosion`、`congo-volcano`、`haiti-earthquake`、
`hawaii-wildfire`、`la_palma-volcano`、`libya-flood`、`morocco-earthquake`、
`turkey-earthquake`。

当前列表的第一个候选是 `bata-explosion_00000014`。也可以用 `--sample-id` 指定另一个
9 事件训练样本；脚本会同时验证它确实在训练列表中。

## 2. Windows 环境检查

推荐使用 Python 3.10。先在 PowerShell 中确认现有环境；这里不启动训练：

```powershell
Set-Location 'D:\BRIGHT-Reproduction\BRIGHT'
python --version
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('build:', torch.version.cuda); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable')"
python -m pip check
```

如尚未创建环境，可使用仓库已记录的兼容版本。不要先安装 CPU 版 PyTorch 后再期待
CUDA 可用：

```powershell
Set-Location 'D:\BRIGHT-Reproduction\BRIGHT'
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r .\requirements-reproduction.txt
python -m pip check
```

如果 PowerShell 的执行策略禁止激活脚本，可不改系统策略，直接把后续命令中的
`python` 换成 `.\.venv\Scripts\python.exe`。

## 3. 运行真实数据烟测

RTX 3050 Ti 建议先用 640×640、batch size 1。这个尺寸与官方训练 crop 一致，比整幅
1024×1024 前向更稳妥：

```powershell
Set-Location 'D:\BRIGHT-Reproduction\BRIGHT'
New-Item -ItemType Directory -Force '.\reports' | Out-Null
python .\tools\smoke_test_real_data.py `
  --dataset-root 'D:\BRIGHT-Reproduction\data' `
  --device cuda `
  --forward-size 640 `
  --report '.\reports\real_data_smoke_test.json'
$LASTEXITCODE
```

不指定 `--sample-id` 时脚本自动选第一个完整候选。显式测试第一个候选的命令为：

```powershell
python .\tools\smoke_test_real_data.py `
  --dataset-root 'D:\BRIGHT-Reproduction\data' `
  --sample-id 'bata-explosion_00000014' `
  --device cuda `
  --forward-size 640 `
  --report '.\reports\real_data_smoke_test.json'
```

若当前 PyTorch 没有 CUDA 支持，可把 `--device cuda` 改为 `--device cpu`；这仍能验证
同一官方模型和真实输入，只是较慢。`--device auto` 会优先使用可用 CUDA。传入
`--forward-size 0` 会尝试整幅 1024×1024 前向，3050 Ti 可能因可用显存不足失败，
并不是本阶段的必要条件。

成功时 JSON 的关键字段应满足：

- `status` 为 `passed`，`sample.paper_nine_event_member` 为 `true`；
- `raw_tiff` 保留三个文件实际的 shape、dtype、最小/最大值和标签唯一值；
- 官方加载器输出的光学/SAR shape 均为 `[3,1024,1024]`，并逐通道匹配源码中的
  归一化公式；
- 默认前向输入为 `[1,6,640,640]`，输出为 `[1,4,640,640]`；
- `forward.output_all_finite` 为 `true`，并记录实际设备和 CUDA 峰值张量显存。

未传 `--checkpoint` 时模型使用 PyTorch 随机初始化，只验证接口和设备兼容性。
`argmax_values_present` 不是实验指标，不能与论文比较。即使加载权重，单个训练样本的
argmax 取值也不是正式评价结果。

## 4. 官方 UNet 权重核查

官方模型记录：<https://zenodo.org/records/15349462>

与当前 `UNet(in_channels=6, num_classes=4)` 结构兼容的标准划分权重为：

| 项目 | 已核实值 |
|---|---|
| 文件名 | `ckpt_UNet_standard_ML_split.pth` |
| 直接下载 | <https://zenodo.org/api/records/15349462/files/ckpt_UNet_standard_ML_split.pth/content> |
| 文件大小 | 124,274,886 bytes（约 118.52 MiB） |
| Zenodo MD5 | `4dc3dc6995a74faaf37d663e6d200037` |
| 格式 | 136 项的 PyTorch `OrderedDict` state dict |
| 首层 | `encoder1.block.0.weight`: `[64,6,3,3]` |
| 末层 | `final_conv.weight`: `[4,64,1,1]` |

以上元数据和权重结构已于 2026-10-20 从 Zenodo 记录及实际文件核验。Windows 下载和
校验命令：

```powershell
Set-Location 'D:\BRIGHT-Reproduction\BRIGHT'
New-Item -ItemType Directory -Force '.\weights' | Out-Null
Invoke-WebRequest `
  -Uri 'https://zenodo.org/api/records/15349462/files/ckpt_UNet_standard_ML_split.pth/content' `
  -OutFile '.\weights\ckpt_UNet_standard_ML_split.pth'
Get-FileHash '.\weights\ckpt_UNet_standard_ML_split.pth' -Algorithm MD5
```

加载官方权重做同一次接口烟测：

```powershell
python .\tools\smoke_test_real_data.py `
  --dataset-root 'D:\BRIGHT-Reproduction\data' `
  --device cuda `
  --forward-size 640 `
  --checkpoint '.\weights\ckpt_UNet_standard_ML_split.pth' `
  --report '.\reports\real_data_smoke_test_official_weight.json'
```

需要严格区分两件事：这个文件可以严格加载到当前官方 UNet，但 Zenodo 没有随权重
提供训练样本清单、划分 checksum 或“仅论文 9 事件”的说明。该权重在 2025-05-07
进入官方记录，而仓库 2025-05-05 的 `standard_ML` 列表已包含 14 个事件。因此不能把
它表述成已经证实的“9 事件子集专用权重”。Zenodo 当前也没有另一个明确标记为
9-event standard split 的 UNet 文件。各个 `ckpt_UNet_cross_event_zeroshot_*.pth` 是
留一事件的跨事件实验，不等价于论文 9 事件标准划分权重。

结论是：该标准权重适合检查模型结构、checkpoint 加载和推理链路；严格复现
2229/349/661 的论文 9 事件训练结果仍应按派生的 9 事件列表自行训练，不能用该权重
代替并声称训练设置一致。

## 5. Windows 与官方 Linux 入口的兼容点

官方训练/推理脚本顶部追加了作者 Linux 机器的绝对 `sys.path`，并在模型和张量上
直接调用 `.cuda()`；命令行中的 `--cuda` 实际没有控制这些调用。Windows 路径本身可
被 Dataset 的 `os.path.join` 正确处理，但直接运行官方入口时容易遇到导入路径错误，
CPU 环境也必然失败。

本阶段的最小兼容方案是使用新增的外部包装脚本：它根据自身位置加入
`bda_benchmark`，用 `.to(device)` 选择设备，并直接调用 Dataset 的单个元素，不启用
DataLoader 多进程。官方 `make_data_loader.py`、`imutils.py`、`UNet.py`、训练脚本和
推理脚本均保持原样。

常见问题：

- `torch.cuda.is_available() is False`：安装 CUDA 版 PyTorch，确认 NVIDIA 驱动可用，
  并避免在同一环境残留 CPU wheel；也可先用 `--device cpu` 完成本阶段验证。
- CUDA out of memory：关闭占用显存的软件后重试 640；或改用 CPU。若只为硬件接口
  排错，可临时用 512 等 16 的倍数，但必须记录这是缩小输入的烟测变体。
- `No module named cv2`：安装 `requirements-reproduction.txt`，其中已补充
  `opencv-python-headless`；虽然本次关闭增强，官方 `imutils.py` 在导入时仍需要 cv2。
- SAR shape 报错：官方要求 TIFF 被读取成 `H×W`，不是 `H×W×1`；不要为通过测试而
  改写数据，先核对解压文件与数据完整性报告。
- checkpoint 出现 missing/unexpected keys：确认下载的是上述标准 UNet 文件且 MD5
  相符，不要静默删除 `module.` 或忽略不匹配层。

完成本地运行后，请保留 `reports\real_data_smoke_test*.json`。在确认真实样本、设备与
可选 checkpoint 都通过之前，不进入 GPU 正式训练配置。
