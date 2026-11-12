"""单层自回归 Transformer Decoder，搭配 encoder.py 运行。

运行：python decoder.py
B：批量；T：Decoder 序列长度；S：Encoder 序列长度；D：特征维度。
采用 Post-LayerNorm；例子使用等长序列，省略 padding mask 和 dropout。
"""

import math

import torch
from torch import nn

from encoder import EncoderLayer


class MultiHeadAttention(nn.Module):
    """分别指定 Q 的来源和 K/V 的来源，同时支持自注意力与交叉注意力。"""

    def __init__(self, d_model=8, num_heads=2):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("特征维度必须能被注意力头数整除")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_layer = nn.Linear(d_model, d_model)
        self.k_layer = nn.Linear(d_model, d_model)
        self.v_layer = nn.Linear(d_model, d_model)
        self.output_layer = nn.Linear(d_model, d_model)

    def forward(self, query_input, key_value_input, mask=None):
        # 自注意力：两个输入都是 Decoder 当前的特征。
        # 交叉注意力：query_input 来自 Decoder，key_value_input 来自 Encoder。
        batch_size, query_len, d_model = query_input.shape
        source_len = key_value_input.shape[1]

        q = self.q_layer(query_input)      # [B, T, D]
        k = self.k_layer(key_value_input)  # [B, S, D]
        v = self.v_layer(key_value_input)  # [B, S, D]

        q = q.reshape(batch_size, query_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, source_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, source_len, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)  # [B, 头数, T, head_dim]
        k = k.transpose(1, 2)  # [B, 头数, S, head_dim]
        v = v.transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # scores：[B, 头数, T, S]。mask 中 True 表示禁止读取。
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        context = weights @ v  # [B, 头数, T, head_dim]
        context = context.transpose(1, 2).reshape(batch_size, query_len, d_model)
        return self.output_layer(context), weights


class DecoderLayer(nn.Module):
    def __init__(self, d_model=8, num_heads=2, hidden_dim=16):
        super().__init__()
        # 两个注意力模块各有自己的可训练参数。
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, x, memory):
        # x：[B, T, D]；memory 是 Encoder 输出：[B, S, D]。
        seq_len = x.shape[1]
        # 对角线上方为 True：当前位置不能读取未来位置。
        # 例如 T=3：[[False, True, True],
        #            [False, False, True],
        #            [False, False, False]]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
            diagonal=1,
        )

        # 1. 带掩码的自注意力：Q、K、V 都由 x 产生。
        self_output, self_weights = self.self_attention(x, x, mask=causal_mask)
        x = self.norm1(x + self_output)

        # 2. 交叉注意力：Q 由更新后的 x 产生，K、V 由 memory 产生。
        # Encoder 输入可被完整读取，这里不加因果掩码。
        cross_output, cross_weights = self.cross_attention(x, memory)
        x = self.norm2(x + cross_output)

        # 3. FFN：分别加工每个位置的特征。
        ffn_output = self.ffn(x)
        x = self.norm3(x + ffn_output)
        return x, self_weights, cross_weights


def main():
    torch.manual_seed(42)
    # 演示任务：将数字序列倒序；10 号 token 作为开始符 BOS。
    source = torch.tensor([[2, 8, 1, 4], [7, 3, 9, 2]])  # [B=2, S=4]
    target = torch.tensor([[4, 1, 8, 2], [2, 9, 3, 7]])
    # Teacher forcing：训练时将正确答案右移一位，作为 Decoder 输入。
    decoder_tokens = torch.tensor([[10, 4, 1, 8], [10, 2, 9, 3]])

    d_model, vocab_size = 8, 11
    token_embedding = nn.Embedding(vocab_size, d_model)
    position_embedding = nn.Embedding(4, d_model)
    encoder = EncoderLayer(d_model, num_heads=2, hidden_dim=16)
    decoder = DecoderLayer(d_model, num_heads=2, hidden_dim=16)
    output_head = nn.Linear(d_model, vocab_size)

    positions = torch.arange(4)
    source_features = token_embedding(source) + position_embedding(positions)
    memory, _ = encoder(source_features)

    x = token_embedding(decoder_tokens) + position_embedding(positions)
    output, self_weights, cross_weights = decoder(x, memory)
    logits = output_head(output)  # [B, T, 11]：每个位置预测下一个 token。

    # CrossEntropyLoss 接收原始 logits，内部完成所需计算。
    loss = nn.functional.cross_entropy(
        logits.reshape(-1, vocab_size), target.reshape(-1)
    )
    loss.backward()  # 展示梯度可从任务损失传回 Encoder 和 Decoder。

    print("Encoder 输出：", tuple(memory.shape))
    print("Decoder 输出：", tuple(output.shape))
    print("分类分数 logits：", tuple(logits.shape))
    print("自注意力权重 [B, 头数, T, T]：\n", self_weights[0, 0].detach())
    print("交叉注意力权重 [B, 头数, T, S]：\n", cross_weights[0, 0].detach())
    print("本次损失：", loss.item())
    print("本次预测编号：\n", logits.argmax(dim=-1))
    # 参数随机初始化；学习倒序任务时，需增加优化器并反复训练。
    # 自回归推理从 BOS 开始，逐步追加预测 token，再调用 Decoder。


if __name__ == "__main__":
    main()
