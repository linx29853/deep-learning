# 最小 Transformer

使用基础 PyTorch 层搭建 Encoder–Decoder Transformer，包含多头注意力、因果掩码、交叉注意力、残差连接、LayerNorm 和 FFN。

## 阅读顺序

1. [图文笔记](notes.md)：Token、Embedding、注意力与整体结构。
2. [encoder.py](encoder.py)：单层 Encoder 和形状示例。
3. [decoder.py](decoder.py)：因果自注意力、交叉注意力与反向传播示例。
4. [minimal_transformer.py](minimal_transformer.py)：完整模型、训练循环和自回归生成。

## 运行

以下命令在仓库根目录执行：

```bash
pip install -r requirements.txt
python transformer/encoder.py
python transformer/decoder.py
python transformer/minimal_transformer.py
# 使用可用的 CUDA 设备训练
python transformer/minimal_transformer.py --device cuda
```

## 结构与任务

单层 Encoder、单层 Decoder，特征维度 32，4 个注意力头，FFN 隐藏维度 64，共 22,444 个参数。数字编号 0～9，BOS=10，EOS=11。输入固定为四位数字，输出为倒序数字加结束符。

训练采用 teacher forcing：目标右移一位作为 Decoder 输入，交叉熵监督每个输出位置。生成从 BOS 开始，只输入源序列，逐步追加预测 token，遇到 EOS 后结束。

训练和推理使用可学习的位置嵌入与 Post-LayerNorm 排列。各条序列等长，直接使用因果掩码。

## 运行结果

[training.log](training.log) 保存 1500 步训练记录。Adam 学习率 0.001，批量大小 128，随机种子 42；损失由 2.6188 降至 0.0006。

独立随机流采样 1000 条评估序列，自回归整段正确率（含 EOS）为 100%。所有四位数字序列共 10,000 种，评估样本可能与训练样本重合。

模型权重运行后保存至 `transformer/minimal_transformer.pt`。`.gitignore` 排除权重文件，重新训练可生成。

## 参考

[Attention Is All You Need](https://arxiv.org/abs/1706.03762)
