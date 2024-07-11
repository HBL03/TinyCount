import torch
import torch.nn as nn
import torch.nn.functional as F
from misc.utils import initialize_weights

# 2D Convolution + BN + Activation Function
class Conv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation, dilation=1):
        super(Conv2d, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            activation
        )

class TransposedConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation, padding=0):
        super(TransposedConv2d, self).__init__(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            activation
        )

# Depthwise Convolution + BN + Activation Function
class DWConv2d(nn.Sequential):
    def __init__(self, in_channels, activation, kernel_size=3, padding=0, stride=1, dilation=1):
        super(DWConv2d, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, groups=in_channels,
                      padding=padding, stride=stride, dilation=dilation , bias=False),
            nn.BatchNorm2d(in_channels),
            activation
        )

# Pointwise Convolution + BN + Activation Function
class PWConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, activation, kernel_size=1, padding=0, stride=1, dilation=1):
        super(PWConv2d, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      padding=padding, stride=stride, dilation=dilation , bias=False),
            nn.BatchNorm2d(out_channels),
            activation
        )

class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation, dilation=1, expand_ratio=1, se=True):
        super(InvertedResidualBlock, self).__init__()

        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            # Expansion layer
            layers.append(PWConv2d(in_channels, hidden_dim, activation=activation))
        layers.extend([
            DWConv2d(hidden_dim, kernel_size=kernel_size, stride=stride, padding=padding, activation=activation, dilation=dilation),
            PWConv2d(hidden_dim, out_channels, activation=activation)
        ])
        self.layers = nn.Sequential(*layers)
        self.se = se
        if self.se is True:
            self.seblock = SEBlock(out_channels)

    def forward(self, x):
        if self.use_res_connect:
            x = x + self.layers(x)
            if self.se is True:
                x = self.seblock(x) * x
            return x
        else:
            x = self.layers(x)
            if self.se is True:
                x = self.seblock(x) * x
            return x

class SEBlock(nn.Module):
    def __init__(self, in_channels, r=4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1,1))
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // r),
            nn.ReLU(),
            nn.Linear(in_channels // r, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.squeeze(x)
        x = x.view(x.size(0), -1) 
        x = self.excitation(x)
        x = x.view(x.size(0), x.size(1), 1, 1)
        return x

class TinyCount(nn.Module):
    def __init__(self):
        super(TinyCount, self).__init__()

        # Feature Extract Module
        self.front_layer_0 = nn.Sequential(
            Conv2d(3, 4, kernel_size=3, stride=1, padding=1, activation=nn.ReLU6(inplace=True)),
            Conv2d(4, 8, kernel_size=3, stride=1, padding=1, activation=nn.ReLU6(inplace=True)),
        )
        self.front_layer_1 = nn.Sequential(
            InvertedResidualBlock(8, 8, stride=2, kernel_size=3, padding=1, expand_ratio=1, activation=nn.ReLU6(inplace=True), se=True),
            InvertedResidualBlock(8, 16, stride=1, kernel_size=3, padding=1, expand_ratio=1, activation=nn.ReLU6(inplace=True), se=True),
        )
        self.front_layer_2 = nn.Sequential(
            InvertedResidualBlock(16, 16, stride=2, kernel_size=3, padding=1, expand_ratio=2, activation=nn.ReLU6(inplace=True), se=True),
            InvertedResidualBlock(16, 32, stride=1, kernel_size=3, padding=1, expand_ratio=2, activation=nn.ReLU6(inplace=True), se=True),
        )
        self.front_layer_3 = nn.Sequential(
            InvertedResidualBlock(32, 32, stride=2, kernel_size=3, padding=1, expand_ratio=3, activation=nn.ReLU6(inplace=True), se=True),
            InvertedResidualBlock(32, 48, stride=1, kernel_size=3, padding=1, expand_ratio=3, activation=nn.ReLU6(inplace=True), se=True),
            InvertedResidualBlock(48, 48, stride=1, kernel_size=3, padding=1, expand_ratio=3, activation=nn.ReLU6(inplace=True), se=True),
        )

        # Upsampling Module
        self.back_layer_3 = nn.Sequential(
            InvertedResidualBlock(48, 32, stride=1, kernel_size=3, padding=1, expand_ratio=1, activation=nn.ReLU6(inplace=True), se=True),
            TransposedConv2d(32, 32, kernel_size=2, stride=2, activation=nn.ReLU6(inplace=True)),
        )
        self.back_layer_2 = nn.Sequential(
            InvertedResidualBlock(32, 16, stride=1, kernel_size=3, padding=1, expand_ratio=1, activation=nn.ReLU6(inplace=True), se=True),
            TransposedConv2d(16, 16, kernel_size=2, stride=2, activation=nn.ReLU6(inplace=True)),
        )
        self.back_layer_1 = nn.Sequential(
            InvertedResidualBlock(16, 8, stride=1, kernel_size=3, padding=1, expand_ratio=1, activation=nn.ReLU6(inplace=True), se=True),
            TransposedConv2d(8, 8, kernel_size=2, stride=2, activation=nn.ReLU6(inplace=True)),
        )
        self.back_layer_0 = nn.Sequential(
            Conv2d(8, 4, kernel_size=3, stride=1, padding=1, activation=nn.ReLU6(inplace=True)),
            PWConv2d(4, 1, activation=nn.ReLU6(inplace=True)),
        )

        # Scale Perception Module
        self.scale_perception_layer_0 = PWConv2d(48, 24, activation=nn.ReLU6(inplace=True))
        self.scale_perception_layer_1 = nn.Sequential(
            InvertedResidualBlock(24, 24, stride=1, kernel_size=3, padding=1, expand_ratio=1, dilation=1, activation=nn.ReLU6(inplace=True)),
            InvertedResidualBlock(24, 24, stride=1, kernel_size=3, padding=1, expand_ratio=1, dilation=1, activation=nn.ReLU6(inplace=True)),
            InvertedResidualBlock(24, 24, stride=1, kernel_size=3, padding=1, expand_ratio=1, dilation=1, activation=nn.ReLU6(inplace=True)),
        )
        self.scale_perception_layer_2 = nn.Sequential(
            InvertedResidualBlock(24, 24, stride=1, kernel_size=3, padding=2, expand_ratio=1, dilation=2, activation=nn.ReLU6(inplace=True)),
            InvertedResidualBlock(24, 24, stride=1, kernel_size=3, padding=2, expand_ratio=1, dilation=2, activation=nn.ReLU6(inplace=True)),
            InvertedResidualBlock(24, 24, stride=1, kernel_size=3, padding=2, expand_ratio=1, dilation=2, activation=nn.ReLU6(inplace=True)),
        )
        self.scale_perception_layer_3 = PWConv2d(24, 48, activation=nn.ReLU6(inplace=True))
        
        initialize_weights(self.modules()) 


    def forward(self, x):
        x = self.front_layer_0(x)
        x = self.front_layer_1(x)
        x = self.front_layer_2(x)
        x = self.front_layer_3(x)

        x = self.scale_perception_layer_0(x)
        x = self.scale_perception_layer_1(x)
        x = self.scale_perception_layer_2(x)
        x = self.scale_perception_layer_3(x)

        x = self.back_layer_3(x)
        x = self.back_layer_2(x)
        x = self.back_layer_1(x)
        x = self.back_layer_0(x)
        return x