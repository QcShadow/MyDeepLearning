from torch import nn
from torch.nn import Conv2d, MaxPool2d, Linear, Flatten, ReLU, Dropout
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

# 自定义神经网络模型
class SENet(nn.Module):
    def __init__(self):
        super(SENet, self).__init__()
        self.conv1 = Conv2d(3, 64, 5, padding=2)
        self.maxpool1 = MaxPool2d(2)
        self.se1 = SEBlock(64)

        self.conv2 = Conv2d(64, 128, 3, padding=1)
        self.maxpool2 = MaxPool2d(2)
        self.se2 = SEBlock(128)

        self.conv3 = Conv2d(128, 128, 3, padding=1)
        self.maxpool3 = MaxPool2d(2)
        self.se3 = SEBlock(128)


        self.flatten = Flatten()
        self.linear1 = Linear(128 * 28 * 28,1024)  # 调整输入特征大小
        self.relu = ReLU()

        self.linear2 = Linear(1024, 4)


    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.se1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.se2(x)

        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.se3(x)

        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
