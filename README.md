# Deep Learning

深度学习实验代码、学习笔记、配图与运行记录。代码使用 PyTorch，按 CNN 与 Transformer 两个主题组织。

## 主题入口

| 主题 | 代码与运行说明 | 图文笔记 | 主要内容 |
|---|---|---|---|
| CNN | [cnn/README.md](https://github.com/linx29853/deep-learning/blob/cnn/cnn/README.md) | [CNN 笔记](https://github.com/linx29853/deep-learning/blob/cnn/cnn/notes.md) | 张量、卷积、CIFAR-10 分类、数据增强实验 |
| Transformer | [transformer/README.md](https://github.com/linx29853/deep-learning/blob/transformer/transformer/README.md) | [Transformer 笔记](https://github.com/linx29853/deep-learning/blob/transformer/transformer/notes.md) | Embedding、多头注意力、Encoder、Decoder、序列生成 |

## 分支与目录

- `main`：完整合集，包含两个主题。
- `cnn`：CNN 主题快照，代码在 `cnn/`。
- `transformer`：Transformer 主题快照，代码在 `transformer/`。

主题快照按目录组织，拉取后仍使用相同的运行路径。后续主题更新先整合进 main，再同步对应主题快照。

```bash
# 获取完整合集
git clone https://github.com/linx29853/deep-learning.git

# 单独获取 CNN 主题
git clone --single-branch --branch cnn https://github.com/linx29853/deep-learning.git deep-learning-cnn

# 单独获取 Transformer 主题
git clone --single-branch --branch transformer https://github.com/linx29853/deep-learning.git deep-learning-transformer
```

已有克隆切换主题：

```bash
git fetch origin
git switch cnn
# 或 git switch transformer
# 回到完整合集：git switch main
```

## 环境与运行

验证环境为 Python 3.10、PyTorch 2.7.1、torchvision 0.22.1。推荐创建独立 Python 环境后安装依赖。

```bash
pip install -r requirements.txt

# main 或 cnn 分支：CNN 前向接口检查
python cnn/train.py --check

# main 或 transformer 分支：运行 Encoder、Decoder 示例
python transformer/encoder.py
python transformer/decoder.py

# main 或 transformer 分支：训练最小 Transformer
python transformer/minimal_transformer.py
```

CNN 的 Docker 环境和 CIFAR-10 完整训练命令见主题 README。数据集缓存、模型权重及运行时实验目录由 .gitignore 排除；固定结果记录、配图与 GIF 随代码保存。

## 实际运行记录

- CNN：初始一轮实验测试准确率 51.9%，配置见 [基线记录](https://github.com/linx29853/deep-learning/blob/cnn/cnn/BASELINE.md)。
- Transformer：22,444 个参数，数字倒序任务训练 1500 步，损失从 2.6188 降至 0.0006；随机采样 1000 条序列自回归整段正确率为 100%。输入空间共 10,000 种，采样评估与训练序列可能重合。

## 来源与历史

CNN 项目来源于 [cifar10-CNN](https://github.com/linx29853/cifar10-CNN)，原仓库保留。完整原始提交另存于 [Git bundle](history/README.md)，可恢复此前的提交历史。官方例程源码保留原有注释和来源链接。

图文笔记采用 Markdown 与相对图片路径，可使用 Typora 或 GitHub 阅读。
