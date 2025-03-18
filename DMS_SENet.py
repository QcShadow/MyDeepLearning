import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Linear, Flatten, ReLU, Dropout, Softmax


# SE 模块 通道注意力机制
class SEBlock(nn.Module):
    def __init__(self, in_channels,reduction=16):
        super(SEBlock, self).__init__()
        #全局平均池化 压缩至一维
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # 进行全局平均池化
        y = self.avg_pool(x).view(b, c)
        # 通过全连接层得到通道注意力权重
        y = self.fc(y).view(b, c, 1, 1)
        # 将注意力权重应用到输入特征图上
        return x * y.expand_as(x)



class DMS_SE(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(DMS_SE, self).__init__()
        self.branch3x3 = Conv2d(in_planes, in_planes, 3, padding=1)
        self.branch5x5 = Conv2d(in_planes, in_planes, 5, padding=2)
        self.branch_dil = Conv2d(in_planes, in_planes, 3, padding=2, dilation=2)

        self.ca3x3 = SEBlock(in_planes, reduction)
        self.ca5x5 = SEBlock(in_planes, reduction)
        self.ca_dil = SEBlock(in_planes, reduction)

        self.weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv2d(in_planes * 3, 3, 1),
            Softmax(dim=1)
        )

    def forward(self, x):
        b3 = self.ca3x3(self.branch3x3(x))
        b5 = self.ca5x5(self.branch5x5(x))
        bd = self.ca_dil(self.branch_dil(x))
        concat = torch.cat([b3, b5, bd], dim=1)
        weight = self.weight(concat)
        return weight[:, 0:1] * b3 + weight[:, 1:2] * b5 + weight[:, 2:3] * bd

class DMS_SENet(nn.Module):
    def __init__(self):
        super(DMS_SENet, self).__init__()
        self.conv1 = Conv2d(3, 64, 5, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool1 = MaxPool2d(2)
        self.dms_se1 = DMS_SE(64)

        self.conv2 = Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.maxpool2 = MaxPool2d(2)
        self.dms_se2 = DMS_SE(128)

        self.conv3 = Conv2d(128, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.maxpool3 = MaxPool2d(2)
        self.dms_se3 = DMS_SE(128)


        self.flatten = Flatten()
        self.linear1 = Linear(128 * 28 * 28,1024)  # 调整输入特征大小
        self.relu = ReLU()

        self.linear2 = Linear(1024, 4)


    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.dms_se1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.dms_se2(x)

        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.dms_se3(x)

        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x