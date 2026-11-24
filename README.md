# TRON 靓号地址生成器

用 NVIDIA GPU 在本机搜索指定尾号的 TRON 地址。找到候选后，脚本会用独立的 Python CPU 实现从私钥重新推导地址，并复核尾号。

## 快速开始

需要 64 位 Python 3.10+、支持 CUDA 的 NVIDIA GPU 和兼容的 NVIDIA 驱动。Windows 11、Python 3.13、RTX 5060 Ti 和 CUDA 12.9 已实测。RTX 50 系需要支持 Blackwell 的环境，使用 CUDA 12 时请准备 CUDA 12.8 或更新的 12.x NVRTC。CuPy 官方的 `[ctk]` 安装方式可同时安装所需 CUDA 组件。 [CuPy 安装文档](https://docs.cupy.dev/en/stable/install.html) · [NVIDIA Blackwell 兼容性说明](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/)

在本项目文件夹打开 PowerShell，运行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install "cupy-cuda12x[ctk]"
.\.venv\Scripts\python.exe .\tron_vanity_gpu.py --selftest
```

自检全部通过后，可先测速，再用短尾号试跑。

```powershell
.\.venv\Scripts\python.exe .\tron_vanity_gpu.py 8888 --bench 10
.\.venv\Scripts\python.exe .\tron_vanity_gpu.py 8888
```

完整的安装、参数说明、等待时间估算和常见报错见 [上手教程](GUIDE.md)。

## 支持的搜索规则

- 精确匹配 1–10 位后缀，区分大小写。Base58 排除了 `0`、大写 `O`、大写 `I` 和小写 `l`。
- `-n 3` 搜索 3 个地址。每次命中后使用新的随机起点。
- `--bench` 测吞吐量，`--tune` 实测并缓存 GPU 参数，`--selftest` 校验 CPU 和 GPU 计算结果。

## 实测参考

RTX 5060 Ti 开启调优缓存后的 8 秒复测吞吐量如下。数值按候选地址计，实际速度随设备、驱动和负载变化。

| 后缀长度 | 样例 | 吞吐量 |
|---|---|---:|
| 4 位 | `8888` | 543.5 MH/s |
| 6 位 | `888888` | 540.3 MH/s |
| 8 位 | `88888888` | 703.9 MH/s |

## 私钥输出

正式搜索命中时，脚本会把**私钥明文**显示在终端，并追加到脚本旁的 `tron_vanity_found.txt`。请在可信本机运行，避免直播、录屏、共享终端或把结果文件同步到云端。收款时只分享公开地址。若私钥已经外泄，应按已泄露处理。 [TronLink 私钥说明](https://support.tronlink.org/hc/en-us/articles/17663186047385-What-are-private-key-mnemonic-keystore-and-password)

内置自检验证计算结果。这个项目的密码学实现未经独立安全审计。

## 许可证

本项目使用 [MIT 许可证](LICENSE)。
