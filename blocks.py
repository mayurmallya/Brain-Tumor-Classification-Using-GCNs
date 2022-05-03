import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_features, out_features, num, pool=True):
        super(ConvBlock, self).__init__()
        layers = []
        layers.append(nn.Conv3d(in_channels=in_features, out_channels=out_features, kernel_size=3, stride=1, padding=1, bias=True))
        layers.append(nn.BatchNorm3d(out_features, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))
        for i in range(1, num):
            layers.append(nn.Conv3d(in_channels=out_features, out_channels=out_features, kernel_size=3, stride=1, padding=1, bias=True))
            layers.append(nn.BatchNorm3d(out_features, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
        if pool:
            layers.append(nn.MaxPool3d(kernel_size=2, stride=2))
        self.op = nn.Sequential(*layers)
    def forward(self, x):
        return self.op(x)

