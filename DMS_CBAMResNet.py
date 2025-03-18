import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Linear, Flatten, ReLU, Dropout, Softmax


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=None):
        super(ChannelAttention, self).__init__()
        if reduction is None:
            reduction = 16 if in_planes >= 256 else 8
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return self.sigmoid(out) * x

class CBAM(nn.Module):
    def __init__(self, in_planes, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

class DMS_CBAM(nn.Module):
    def __init__(self,in_channels):
        super(DMS_CBAM, self).__init__()
        self.branch3x3 = nn.Sequential(
            Conv2d(in_channels, in_channels, 3, padding=1,groups=in_channels),
            Conv2d(in_channels, in_channels, 1)
        )
        self.branch5x5 = nn.Sequential(
            Conv2d(in_channels, in_channels, 5, padding=2, groups=in_channels),
            Conv2d(in_channels, in_channels, 1)
        )
        self.branch_dil = nn.Sequential(
            Conv2d(in_channels, in_channels, 3, padding=2, dilation=2,groups=in_channels),
            Conv2d(in_channels, in_channels, 1)
        )

        self.cbam3x3 = CBAM(in_channels, reduction=8)
        self.cbam5x5 = CBAM(in_channels, reduction=16)
        self.cbam_dil = CBAM(in_channels, reduction=16)

        self.weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv2d(in_channels * 3, 32, 1),
            ReLU(),
            Conv2d(32, 3, 1),
            Softmax(dim=1)
        )

    def forward(self, x):
        identity = x
        b3 = self.cbam3x3(self.branch3x3(x))
        b5 = self.cbam5x5(self.branch5x5(x))
        bd = self.cbam_dil(self.branch_dil(x))
        concat = torch.cat([b3, b5, bd], dim=1)
        weight = self.weight(concat)
        out = weight[:, 0:1] * b3 + weight[:, 1:2] * b5 + weight[:, 2:3] * bd
        return out+identity


class DMS_CBAMResNet(nn.Module):
    def __init__(self):
        super(DMS_CBAMResNet, self).__init__()
        self.conv1 = Conv2d(3, 64, 5, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool1 = MaxPool2d(2)
        self.dms_cbam1 = DMS_CBAM(64)

        self.conv2 = Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.maxpool2 = MaxPool2d(2)
        self.dms_cbam2 = DMS_CBAM(128)

        self.conv3 = Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.maxpool3 = MaxPool2d(2)
        self.dms_cbam3 = DMS_CBAM(256)

        self.globalpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = Flatten()
        self.linear1 = Linear(256, 1024)
        self.relu = ReLU()
        # self.dropout = Dropout(0.5)
        self.linear2 = Linear(1024, 4)

    def forward(self, x):
        # x1 = self.maxpool1(self.bn1(self.conv1(x)))  # 保存浅层特征
        # x = self.dms_cbam1(x1)
        #
        # x2 = self.maxpool2(self.bn2(self.conv2(x)))  # 保存中层特征
        # x = self.dms_cbam2(x2)
        #
        # x3 = self.maxpool3(self.bn3(self.conv3(x)))  # 保存深层特征
        # x = self.dms_cbam3(x3)
        #
        # # 特征融合（示例：相加融合）
        # fused_feature = x1 + x2 + x3  # 需调整通道数一致

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool1(x)
        x = self.dms_cbam1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.maxpool2(x)
        x = self.dms_cbam2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.maxpool3(x)
        x = self.dms_cbam3(x)

        x = self.globalpool(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

