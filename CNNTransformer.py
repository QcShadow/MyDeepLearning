import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Linear, Flatten, ReLU, Dropout, Softmax
from torchvision import models


class Transformer(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, num_layers):
        super(Transformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self,x):
        x = self.transformer_encoder(x)
        return x


class CNNTransformer(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, num_layers):
        super(CNNTransformer, self).__init__()
        self.cnn = models.resnet18(pretrained=True)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-2])
        self.transformer = Transformer(input_dim, num_heads, hidden_dim, num_layers)
        self.fc = Linear(input_dim, 4)

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()
        x = x.view(b, c, -1).permute(2, 0, 1)
        x = self.transformer(x)
        x = x[0]
        x = self.fc(x)
        return x
