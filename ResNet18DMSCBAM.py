import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Linear, Flatten, ReLU, Dropout, Softmax
from torchvision.models import resnet18


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

class DMS_CBAM(nn.Module):
    def __init__(self,in_channels, reduction=16):
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

        self.cbam3x3 = CBAM(in_channels, reduction)
        self.cbam5x5 = CBAM(in_channels, reduction)
        self.cbam_dil = CBAM(in_channels, reduction)

        self.weight = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            # Conv2d(in_channels * 3, 3, 1),
            # Softmax(dim=1)
            nn.AdaptiveAvgPool2d(1),
            Conv2d(in_channels * 3, 32, 1),
            ReLU(),
            Conv2d(32, 3, 1),
            Softmax(dim=1)
        )

    def forward(self, x):
        # identity = x
        b3 = self.cbam3x3(self.branch3x3(x))
        b5 = self.cbam5x5(self.branch5x5(x))
        bd = self.cbam_dil(self.branch_dil(x))
        concat = torch.cat([b3, b5, bd], dim=1)
        weight = self.weight(concat)
        out = weight[:, 0:1] * b3 + weight[:, 1:2] * b5 + weight[:, 2:3] * bd
        return out

class ResNet18DMSCBAM(nn.Module):
    def __init__(self, num_classes=4, reduction=16):
        super(ResNet18DMSCBAM, self).__init__()
        self.resnet = resnet18(weights=None)
        self.dms_cbam1 = DMS_CBAM(64, reduction)
        self.dms_cbam2 = DMS_CBAM(128, reduction)
        self.dms_cbam3 = DMS_CBAM(256, reduction)
        self.dms_cbam4= DMS_CBAM(512, reduction)

        self.resnet.fc = Linear(512, num_classes)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.dms_cbam1(x)
        x = self.resnet.layer2(x)
        x = self.dms_cbam2(x)
        x = self.resnet.layer3(x)
        x = self.dms_cbam3(x)
        x = self.resnet.layer4(x)
        x = self.dms_cbam4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)
        return x
