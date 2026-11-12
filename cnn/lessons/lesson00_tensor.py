"""张量基础：从一个数，到灰度图，再到一批 RGB 图。"""

import torch

scalar = torch.tensor(7.0)
vector = torch.tensor([2.0, 5.0, 8.0])
image = torch.tensor([
    [0, 0, 0, 0],
    [0, 255, 255, 0],
    [0, 255, 255, 0],
    [0, 0, 0, 0],
], dtype=torch.float32)

for name, tensor in [("一个数", scalar), ("一排数", vector), ("灰度图", image)]:
    print(name, "shape =", tuple(tensor.shape), "轴数 =", tensor.ndim, "元素数 =", tensor.numel())

print("第 1 行、第 2 列（索引从 0 开始）:", image[1, 2].item())
print("第 1 行:", image[1].tolist())
print("除以 255 后的第 1 行:", (image / 255)[1].tolist())

# 这三个数组表示同一张图片的三个颜色通道。
red = image
green = torch.zeros_like(image)
blue = torch.zeros_like(image)
rgb = torch.stack([red, green, blue], dim=0)
print("RGB 图片 shape:", tuple(rgb.shape))
print("一个红色像素的 RGB 值:", rgb[:, 1, 1].tolist())

# 演示批量轴：把同一张图片复制成两份，仅用于观察形状。
batch = torch.stack([rgb, rgb], dim=0)
print("两张 RGB 图片 shape:", tuple(batch.shape))
print("第二张图的 R 通道、第 1 行第 2 列:", batch[1, 0, 1, 2].item())
print("数据类型:", batch.dtype, "存放设备:", batch.device)

import torch

image = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0],
])

kernel = torch.tensor([
    [1.0,  0.0],
    [0.0, -1.0],
])

output = torch.zeros(2, 2)

for row in range(2):
    for col in range(2):
        patch=image[row:row+2,col:col+2]
        output[row,col]=(patch*kernel).sum()
print(output)