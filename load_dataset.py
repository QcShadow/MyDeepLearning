import os
import random

import numpy as np
# import os
# print("当前工作目录:", os.getcwd())
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision import transforms

# 自定义数据集
class RiceData(Dataset):
    def __init__(self, root_dir, label_dir, transform=None, label=0):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)
        self.transform = transform  # 增加一个transform的参数
        self.label = label


    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path).convert("RGB")
        # transform进行类型转换
        if self.transform:
            img = self.transform(img)
        return img, self.label

    def __len__(self):
        return len(self.img_path)


# 定义椒盐噪声函数
# def add_salt_and_pepper_noise(img, prob=0.02):
#     c, h, w = img.shape
#     noisy_img = img.clone()
#
#     num_pixels = int(prob * h * w)  # 计算需要添加噪声的像素数量
#     for _ in range(num_pixels):
#         y = random.randint(0, h - 1)
#         x = random.randint(0, w - 1)
#         value = 1.0 if random.random() > 0.5 else 0.0  # 随机变白（1.0）或变黑（0.0）
#         noisy_img[:, y, x] = value
#
#     return noisy_img

def add_salt_and_pepper_noise(img, prob=0.02):
    img = np.array(img)
    img_height, img_width = img.shape[:2]
    num_pixels = int(prob * img_height * img_width)

    for _ in range(num_pixels):
        y = random.randint(0, img_height - 1)
        x = random.randint(0, img_width - 1)
        img[y, x] = 255 if random.random() > 0.5 else 0  # 白色或黑色像素

    return Image.fromarray(img)


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=30),
    # transforms.ToTensor(),
    transforms.Lambda(lambda img: add_salt_and_pepper_noise(img, prob=0.02)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 项目所在的的绝对路径
root_dir = "/root/autodl-tmp/project/dataset"

# root_dir = "/root/autodl-tmp/project/redodataset"
# root_dir = "/root/autodl-tmp/project/archive"
# root_dir = "F:\\pycode\\rice\\dataset"

train_root_dir = os.path.join(root_dir, "train")

train_Bacterialblight_dataset = RiceData(train_root_dir, "Bacterialblight", train_transform, label=0)
train_Blast_dataset = RiceData(train_root_dir, "Blast", train_transform, label=1)
train_Brownspot_dataset = RiceData(train_root_dir, "Brownspot", train_transform, label=2)
train_Tungro_dataset = RiceData(train_root_dir, "Tungro", train_transform, label=3)

train_dataset = ConcatDataset([train_Bacterialblight_dataset, train_Blast_dataset, train_Brownspot_dataset, train_Tungro_dataset])

# print(f"训练数据集长度:{len(train_dataset)}")

val_root_dir = os.path.join(root_dir, "val")

val_Bacterialblight_dataset = RiceData(val_root_dir, "Bacterialblight", test_transform, label=0)
val_Blast_dataset = RiceData(val_root_dir, "Blast", test_transform, label=1)
val_Brownspot_dataset = RiceData(val_root_dir, "Brownspot", test_transform, label=2)
val_Tungro_dataset = RiceData(val_root_dir, "Tungro", test_transform, label=3)

val_dataset = ConcatDataset([val_Bacterialblight_dataset, val_Blast_dataset, val_Brownspot_dataset, val_Tungro_dataset])

# print(f"验证数据集长度:{len(val_dataset)}")

test_root_dir = os.path.join(root_dir, "test")

test_Bacterialblight_dataset = RiceData(test_root_dir, "Bacterialblight", test_transform, label=0)
test_Blast_dataset = RiceData(test_root_dir, "Blast", test_transform, label=1)
test_Brownspot_dataset = RiceData(test_root_dir, "Brownspot", test_transform, label=2)
test_Tungro_dataset = RiceData(test_root_dir, "Tungro", test_transform, label=3)

test_dataset = ConcatDataset([test_Bacterialblight_dataset, test_Blast_dataset, test_Brownspot_dataset, test_Tungro_dataset])

# print(f"测试数据集长度:{len(test_dataset)}")
