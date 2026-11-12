"""CIFAR-10 第一课：下载、读取图片，检查标签与张量形状。"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


BASE = Path(__file__).resolve().parent
# 数据缓存在工作目录，避免与练习文件混在一起。
DATA = BASE.parent / "data"
CLASS_NAMES = ["飞机", "汽车", "鸟", "猫", "鹿", "狗", "青蛙", "马", "船", "卡车"]


def main():
    # ToTensor 把 PIL 图片转换为 (C, H, W) 浮点张量，像素从 0～255 转为 0～1。
    train_data = datasets.CIFAR10(
        root=DATA, train=True, download=True, transform=transforms.ToTensor()
    )
    test_data = datasets.CIFAR10(
        root=DATA, train=False, download=True, transform=transforms.ToTensor()
    )

    image, label = train_data[0]
    print("训练图片数：", len(train_data))
    print("测试图片数：", len(test_data))
    print("第一张图片 shape：", tuple(image.shape))
    print("数据类型：", image.dtype)
    print("像素范围：", image.min().item(), "～", image.max().item())
    print("标签：", label, CLASS_NAMES[label])

    # DataLoader 将单张图片组织成一批；此处只取一批查看，不进行训练。
    loader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=0)
    images, labels = next(iter(loader))
    print("一批图片 shape：", tuple(images.shape))
    print("一批标签 shape：", tuple(labels.shape))
    print("一批标签：", labels.tolist())

    # 每类选 3 张真实样本，按固定顺序展示，便于观察类别编号。
    sheet = Image.new("RGB", (1000, 610), "#f7f9fc")
    draw = ImageDraw.Draw(sheet)
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    title_font = ImageFont.truetype(font_path, 25)
    label_font = ImageFont.truetype(font_path, 18)
    draw.text((20, 12), "CIFAR-10：10 个类别，每类 3 张训练样本（32×32 放大）", font=title_font, fill="#182638")
    counts = [0] * 10
    for array, target in zip(train_data.data, train_data.targets):
        if counts[target] >= 3:
            continue
        col, row = target % 5, target // 5
        x, y = col * 200 + 15, row * 270 + 62
        if counts[target] == 0:
            draw.text((x, y), f"{target}：{CLASS_NAMES[target]}", font=label_font, fill="#182638")
        # 放大仅用于查看，训练时仍使用原始 32×32。
        # 保持正方形：每类横向放两张、下一行放一张。
        thumb = Image.fromarray(array).resize((72, 72), Image.Resampling.NEAREST)
        k = counts[target]
        sheet.paste(thumb, (x + (k % 2) * 82, y + 32 + (k // 2) * 82))
        counts[target] += 1
        if all(count == 3 for count in counts):
            break
    preview = BASE / "cifar10-samples.png"
    sheet.save(preview)
    print("样本预览：", preview)


if __name__ == "__main__":
    main()
