import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


########################################
# unit for U-net
########################################
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        x = torch.cat([x2, x1], dim=1)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


#######################################################
#  for PSPNet
#######################################################
class BackNet(object):
    def __init__(self, model, pretrained=True):
        # using dilated filter
        if model == "resnet50":
            self.base_model = models.resnet50(replace_stride_with_dilation=[False, 2, 4], pretrained=pretrained)
        if model == "resnet101":
            self.base_model = models.resnet101(replace_stride_with_dilation=[False, 2, 4], pretrained=pretrained)

    def back(self):
        return self.base_model


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_dim, reduction_dim, setting, lstm=False):
        super(PyramidPoolingModule, self).__init__()
        self.lstm = lstm
        self.features = []
        for s in setting:
            self.features.append(nn.Sequential(
                    nn.AdaptiveAvgPool2d(s),
                    nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(reduction_dim, momentum=.95),
                    nn.ReLU(inplace=True)
                ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode="bilinear"))
        out = torch.cat(out, 1)
        return out
