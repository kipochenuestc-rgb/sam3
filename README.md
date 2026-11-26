# SAM 3 本地分割

使用 Meta 官方 SAM 3 在 WSL / Linux 的 NVIDIA GPU 上离线推理。默认分割天空、人、汽车、电瓶车和自行车。

## 安装

需要 Python 3.12、python3-venv 和支持 CUDA 12.8 的 NVIDIA 驱动。在 Ubuntu 24.04 中执行

```bash
sudo apt-get install python3-venv
bash setup_wsl.sh
```

RTX 5060 使用 cu128 版 PyTorch。已在 WSL Ubuntu 24.04 / RTX 5060 Laptop 8GB 上验证。

## 权重

权重通过 Git LFS 管理。首次克隆需安装 Git LFS 并拉取模型

```bash
git lfs install
git lfs pull
python3 restore_weights.py
```

仓库包含 13 个不超过 256MiB 的权重分片，以适配 GitHub LFS 和网络传输。restore_weights.py 流式合并并执行 SHA-256 校验，得到 weights/sam3.pt，安装脚本也会自动执行此步骤。还原时需要额外约 3.45GB 磁盘空间。SAM 3 权重遵循仓库 LICENSE 中的 SAM License。官方来源为 https://huggingface.co/facebook/sam3 ，首次从官方获取需先申请访问权限。代码和权重就位后可断网推理。

## 调用

```bash
# 批量图片，递归扫描目录
.venv/bin/python segment.py test_pic --output results/run_01

# 单张图片
.venv/bin/python segment.py test_pic/000332.jpg --output results/single

# 视频逐帧分割
.venv/bin/python segment.py input.mp4 --output results/video

# 自定义类别及置信度
.venv/bin/python segment.py test_pic --output results/custom --prompts sky person car "electric scooter" bicycle --threshold 0.35
```

Windows 的 D:\slam\sam3 在 WSL 中对应 /mnt/d/slam/sam3。命令中使用 WSL 路径。输出使用新目录。

每张图片对应一个结果子目录，保留原始相对路径和文件扩展名，例如 results/run_01/000332.jpg/。其中 01_mask.png 等为各类别黑白掩膜，labels.png 为单通道类别 ID 图，overlay.jpg 为淡色填充及轮廓预览。summary.json 记录类别、颜色、置信度、实例数及耗时。

默认标签 ID 为 0 背景，1 天空，2 人，3 汽车，4 电瓶车，5 自行车。独立掩膜允许重叠，标签图在重叠处取后处理的类别。视频采用逐帧分割，不生成跟踪 ID。

## 目录

- segment.py 为统一命令行入口，复用官方模型构建和处理器。
- setup_wsl.sh 安装 WSL 所需运行依赖。
- sam3/ 保留官方模型及内部依赖模块，内含 tokenizer 词表。
- weights/ 保存 Git LFS 模型权重。
- restore_weights.py 合并并校验权重分片。
- test_pic/ 为本地测试输入，results/ 为生成结果，均由 Git 忽略。

## 实测与限制

800×640 图片分割天空、人、汽车三类，BF16 推理稳定平均约 0.53 秒每张，首张约 1.33 秒，模型加载约 26 秒，峰值 GPU 分配约 4.97 GiB。以上来自本机实测，五类和包含文件读写的端到端耗时以 summary.json 为准。模型输入由官方处理器缩放为 1008×1008。远处模糊小目标、遮挡和电瓶车与自行车的类别区分可能出现误检或漏检。

官方代码来源 https://github.com/facebookresearch/sam3 。版权及使用条款见 LICENSE。
