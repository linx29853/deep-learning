"""最小 Encoder–Decoder Transformer：学习四位数字序列倒序。

运行：python minimal_transformer.py
复用同目录的 encoder.py、decoder.py；所有模块注册在同一个模型中。
"""

import argparse
from pathlib import Path

import torch
from torch import nn

from encoder import EncoderLayer
from decoder import DecoderLayer

BOS, EOS = 10, 11  # 开始符和结束符；数字 token 为 0～9
VOCAB_SIZE = 12
SEQ_LEN = 4


class MinimalTransformer(nn.Module):
    def __init__(self, d_model=32, num_heads=4, hidden_dim=64):
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE, d_model)
        # 本任务源序列长 4；目标输入含 BOS，最长为 5。
        self.source_position = nn.Embedding(SEQ_LEN, d_model)
        self.target_position = nn.Embedding(SEQ_LEN + 1, d_model)
        self.encoder = EncoderLayer(d_model, num_heads, hidden_dim)
        self.decoder = DecoderLayer(d_model, num_heads, hidden_dim)
        self.output_head = nn.Linear(d_model, VOCAB_SIZE)

    def encode(self, source):
        positions = torch.arange(source.shape[1], device=source.device)
        x = self.token_embedding(source) + self.source_position(positions)
        memory, _ = self.encoder(x)
        return memory

    def decode(self, target_input, memory):
        positions = torch.arange(target_input.shape[1], device=target_input.device)
        x = self.token_embedding(target_input) + self.target_position(positions)
        features, _, _ = self.decoder(x, memory)
        return self.output_head(features)  # [批量, 目标长度, 12]

    def forward(self, source, target_input):
        return self.decode(target_input, self.encode(source))

    @torch.no_grad()
    def generate(self, source):
        """输入只有源序列：从 BOS 开始，每次选取概率最大的下一个 token。"""
        memory = self.encode(source)
        generated = torch.full(
            (source.shape[0], 1), BOS, dtype=torch.long, device=source.device
        )
        finished = torch.zeros(source.shape[0], dtype=torch.bool, device=source.device)
        for _ in range(SEQ_LEN + 1):
            logits = self.decode(generated, memory)
            next_token = logits[:, -1].argmax(dim=-1)
            next_token = torch.where(finished, EOS, next_token)
            generated = torch.cat([generated, next_token[:, None]], dim=1)
            finished = finished | (next_token == EOS)
            if finished.all():
                break
        return generated[:, 1:]  # 返回预测结果，去掉最初的 BOS


def make_batch(batch_size, device, generator=None):
    source = torch.randint(
        0, 10, (batch_size, SEQ_LEN), device=device, generator=generator
    )
    reversed_digits = source.flip(dims=[1])
    bos = torch.full((batch_size, 1), BOS, device=device, dtype=torch.long)
    eos = torch.full((batch_size, 1), EOS, device=device, dtype=torch.long)
    # 输入：[BOS, 4, 1, 8, 2]；监督目标：[4, 1, 8, 2, EOS]。
    target_input = torch.cat([bos, reversed_digits], dim=1)
    target = torch.cat([reversed_digits, eos], dim=1)
    return source, target_input, target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(2)  # 小网络在 CPU 上使用少量线程即可。
    device = torch.device(args.device)
    model = MinimalTransformer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    print("参数量：", sum(p.numel() for p in model.parameters()), flush=True)
    model.train()
    for step in range(1, args.steps + 1):
        source, target_input, target = make_batch(128, device)
        logits = model(source, target_input)
        loss = loss_fn(logits.reshape(-1, VOCAB_SIZE), target.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == 1 or step % 250 == 0:
            print(f"步骤 {step:4d} | 损失 {loss.item():.4f}", flush=True)

    # 独立随机流生成评估样本；完整枚举有 10000 种输入，训练与评估可能重合。
    model.eval()
    evaluation_rng = torch.Generator(device=device).manual_seed(2026)
    source, target_input, target = make_batch(1000, device, evaluation_rng)
    with torch.no_grad():
        logits = model(source, target_input)
        validation_loss = loss_fn(logits.reshape(-1, VOCAB_SIZE), target.reshape(-1))
        predictions = model.generate(source)  # 不传入正确目标，自回归生成。
    exact_accuracy = (
        (predictions == target).all(dim=1).float().mean().item()
        if predictions.shape == target.shape else 0.0
    )
    print(f"评估损失：{validation_loss.item():.4f}")
    print(f"自回归整段正确率（含 EOS）：{exact_accuracy:.1%}")
    print("EOS 编号为 11：")
    for src, pred, truth in zip(source[:5], predictions[:5], target[:5]):
        print(f"输入 {src.tolist()} → 生成 {pred.tolist()} | 目标 {truth.tolist()}")

    checkpoint = Path(__file__).with_name("minimal_transformer.pt")
    torch.save(model.state_dict(), checkpoint)
    print("权重已保存：", checkpoint)


if __name__ == "__main__":
    main()
