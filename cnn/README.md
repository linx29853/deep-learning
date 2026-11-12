# CIFAR-10 TinyCNN

使用 PyTorch 实现的 CIFAR-10 图像分类项目，包含 CNN、交叉熵损失和参数更新逻辑，以及训练、验证和模型保存流程。

## 网络结构

输入为 RGB 图片，形状为 `(B, 3, 32, 32)`。两组卷积、ReLU 和最大池化后，特征形状为 `(B, 32, 8, 8)`；展平为 2048 个特征，经过 `2048 → 64 → 10` 全连接层输出十类原始分数。网络共有 136,874 个可训练参数。

## 环境与运行

验证环境：Python、PyTorch 2.7.1、torchvision 0.22.1。程序自动选择可用的 CUDA 设备或 CPU。

```bash
pip install -r requirements.txt
python train.py --check
python train.py --download --epochs 5
```

`--check` 使用随机张量检查网络输出和损失接口，不更新参数。数据下载至 `data/`，首次训练使用 `--download`。默认批量大小为 64，Adam 学习率为 0.001。


## Docker（CPU）

在仓库根目录构建镜像并检查网络接口：

```bash
docker build -t cifar10-cnn:cpu .
docker run --rm cifar10-cnn:cpu
```

镜像使用 Python 3.11，按 requirements.txt 安装 CPU 版 PyTorch 和 torchvision，无需在宿主机安装 Python 依赖。已在 Ubuntu 22.04.5、x86_64 上完成镜像构建和容器内 `--check` 验证；该检查不执行训练或参数更新。

在 Linux 上训练并将数据和最佳模型保存在当前项目目录：

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  --mount type=bind,source="$(pwd)",target=/app \
  cifar10-cnn:cpu --download --epochs 5
```

挂载后使用当前目录中的 train.py。数据保存至 data/，模型保存至 cifar10-best.pt，容器退出后保留。首次下载数据需要网络连接。该镜像使用 CPU 计算。

.dockerignore 排除数据集、模型权重和本地环境，避免将它们发送到构建上下文。

## 数据与评估

原始 CIFAR-10 训练集划分为 45,000 张训练图片和 5,000 张验证图片，划分种子为 42。原始 10,000 张测试图片用于最终评估。每轮输出训练与验证损失、准确率；验证准确率最高的模型保存为 `cifar10-best.pt`，训练结束后加载该模型进行测试。

## 已执行检查

网络前向接口已验证：`(4, 3, 32, 32) → (4, 10)`，分类损失返回标量张量。随机数据单批训练已验证梯度计算和参数更新。正式分类准确率通过完整数据训练与测试获得。

## 参考

- [CIFAR-10 数据集](https://www.cs.toronto.edu/~kriz/cifar.html)
- [PyTorch 分类入门教程](https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)

## 代码与笔记

- [学习笔记与配图](notes.md)
- [张量基础](lessons/lesson00_tensor.py)
- [CNN 形状变化](lessons/lesson01.py)
- [CIFAR-10 数据与标签](lessons/lesson02_cifar10_data.py)
- [PyTorch 官方例程源码](official_cifar10_tutorial.py)
- [随机水平翻转对照实验](experiments/augmentation/AUGMENTATION_EXPERIMENT.md)

本页命令在 `cnn/` 目录下执行。
