# TRON 靓号地址生成器上手教程

本教程以 Windows PowerShell 和 NVIDIA GPU 为例。测试阶段可以先用 `--selftest` 和 `--bench`，这两个模式不会保存命中的私钥。

## 1. 下载和检查环境

在 GitHub 仓库的 **Code → Download ZIP** 下载并解压。装有 Git 也可以运行：

```powershell
git clone https://github.com/uernamecan/tron-vanity-gpu.git
Set-Location .\tron-vanity-gpu
```

在项目目录打开 PowerShell，确认显卡驱动和 Python 可用。

```powershell
nvidia-smi
python --version
```

需要 64 位 Python 3.10 或更新版本。首次安装依赖需要能访问 Python 软件包索引，运行搜索本身可以离线。

## 2. 安装 CuPy 并自检

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install "cupy-cuda12x[ctk]"
.\.venv\Scripts\python.exe .\tron_vanity_gpu.py --selftest
```

自检先核对 CPU 已知向量，再检查 GPU 的普通后缀和长后缀路径。看到全部通过后再运行搜索。以下命令一直使用同一个虚拟环境里的 Python。

## 3. 测速、搜索和调参

先测 10 秒，确认 GPU 运行正常，并记录搜索吞吐量。这里的 MH/s 是每秒检查的百万个候选地址。

```powershell
.\.venv\Scripts\python.exe .\tron_vanity_gpu.py Rico --bench 10
```

以下是正式搜索示例。运行后可按 Ctrl+C 停止。

```powershell
# 搜以 8888 结尾的一个地址
.\.venv\Scripts\python.exe .\tron_vanity_gpu.py 8888

# 搜以 Rico 结尾的三个地址
.\.venv\Scripts\python.exe .\tron_vanity_gpu.py Rico -n 3
```

如果双击打开脚本，窗口会进入输入后缀的交互模式。测速、调参和自检参数在 PowerShell 启动时传入。

如果要针对当前显卡调参，使用和目标相同长度的后缀。调参完成后，脚本会将最快且通过校验的候选配置写入同目录下的 `tron_vanity_tune.json`。缓存按 GPU 型号和后缀档位保存，档位为 1–5 位、6 位、7–10 位。出现「参数来源：调优缓存」表示配置已加载。

```powershell
.\.venv\Scripts\python.exe .\tron_vanity_gpu.py Rico --tune
.\.venv\Scripts\python.exe .\tron_vanity_gpu.py Rico --bench 10
```

## 4. 估算要跑多久

指定 `k` 位精确后缀，平均约需检查 `58^k` 个候选。用候选数除以实测速率，就能估算平均耗时。下表按 700 MH/s 计算，搜索过程中的速率和单次命中时间会变化。

| 精确后缀长度 | 平均耗时 |
|---|---:|
| 6 位 | 约 54 秒 |
| 7 位 | 约 53 分钟 |
| 8 位 | 约 2.1 天 |
| 9 位 | 约 123 天 |
| 10 位 | 约 19.5 年 |

搜索窗口的「平均约需」是按当前速度估算的等待时间。观察是否在运行，可看「已尝试」计数是否增长。

## 5. 管理搜索结果

默认结果写在脚本旁的 `tron_vanity_found.txt`。如果要改位置，可以先建立专用目录，再传入 `-o`。

```powershell
New-Item -ItemType Directory -Path .\results -Force | Out-Null
.\.venv\Scripts\python.exe .\tron_vanity_gpu.py Rico -o .\results\tron_vanity_found.txt
```

命中时，终端和结果文件都含明文私钥。请将实用私钥备份到受保护的位置，避免把生成结果、截图或终端日志提交到仓库。普通接收转账只需分享公开地址。 [TronLink 私钥说明](https://support.tronlink.org/hc/en-us/articles/17663186047385-What-are-private-key-mnemonic-keystore-and-password)

## 常见问题

| 现象 | 处理方法 |
|---|---|
| `No module named 'cupy'` | 用教程中同一条 `.\.venv\Scripts\python.exe` 路径安装依赖和启动脚本。 |
| `CUDA path could not be detected` | 先运行 `--selftest`。如果自检通过，当前 CuPy 能编译运行本脚本的 GPU 内核。 |
| 找不到 NVRTC 或 CUDA Runtime DLL | 检查 `cupy-cuda12x[ctk]` 是否装全。若仍缺组件，可安装兼容的 CUDA Toolkit 12.x。RTX 50 系请使用 12.8 或更新的 12.x。 |
| 显卡没有被识别 | 先用 `nvidia-smi` 检查驱动和设备。多卡环境可用 `--device 0` 指定 GPU 编号。 |
| 后缀输入报错 | 检查长度和大小写。Base58 排除了 `0`、`O`、`I`、`l`。想搜 9 个大写 A，要输入 `AAAAAAAAA`。 |
| GPU 已知答案自检失败 | 可尝试 `--selftest --no-ptx` 复查通用实现。等自检通过后再搜索，反馈问题时提供 GPU 型号、驱动版本和去掉私钥的报错。 |

[CuPy 安装文档](https://docs.cupy.dev/en/stable/install.html) · [NVIDIA Blackwell 兼容性说明](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/) · [TRON 地址编码](https://developers.tron.network/docs/encoding)
