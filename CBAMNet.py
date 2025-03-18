import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Linear, Flatten, ReLU, Dropout, Softmax


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(ChannelAttention, self).__init__()
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

class CBAMNet(nn.Module):
    def __init__(self):
        super(CBAMNet, self).__init__()
        self.conv1 = Conv2d(3, 64, 5, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool1 = MaxPool2d(2)
        self.cbam1 = CBAM(64)

        self.conv2 = Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.maxpool2 = MaxPool2d(2)
        self.cbam2 = CBAM(128)

        self.conv3 = Conv2d(128, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.maxpool3 = MaxPool2d(2)
        self.cbam3 = CBAM(128)

        self.flatten = Flatten()
        self.linear1 = Linear(128 * 28 * 28, 1024)
        self.relu = ReLU()
        self.linear2 = Linear(1024, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool1(x)
        x = self.cbam1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.maxpool2(x)
        x = self.cbam2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.maxpool3(x)
        x = self.cbam3(x)

        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

