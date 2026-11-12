"""第一课：观察图像经过 CNN 时的张量形状。"""

import torch
from torch import nn


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # RGB 图像有 3 个通道；16 和 32 是网络学习的特征通道数。
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # 两次池化使 32×32 变为 8×8；最终预测目标的 x、y。
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, images):
        features = self.features(images)
        return self.head(features)


if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyCNN().to(device)
    # NCHW：批量大小、颜色通道、高、宽。随机图仅用于检查网络流程。
    images = torch.rand(4, 3, 32, 32, device=device)
    print("运行设备:", device)
    print("输入:", tuple(images.shape))
    with torch.no_grad():
        intermediate = images
        for layer in [*model.features, *model.head]:
            intermediate = layer(intermediate)
            print(f"{layer.__class__.__name__:12s} -> {tuple(intermediate.shape)}")
        predictions = model(images)
    print("第一张图的输出:", predictions[0].cpu().tolist())
    print("随机初始化网络：本例展示各层形状变化。")

