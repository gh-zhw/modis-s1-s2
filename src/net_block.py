import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, bias=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=bias),
            nn.InstanceNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        output = self.conv(x)
        return output


class DeconvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, bias=False):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride, padding, bias=bias),
            nn.InstanceNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        output = self.deconv(x)
        return output


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class ConvSpatialAttentionBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, bias=False):
        super().__init__()
        self.conv = ConvBlock(in_channel, out_channel, kernel_size, stride, padding, bias=bias)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.conv(x)
        output = x * self.sa(x)
        return output


class DeconvChannelAttentionBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, bias=False):
        super().__init__()
        self.deconv = DeconvBlock(in_channel, out_channel, kernel_size, stride, padding, bias=bias)
        self.ca = ChannelAttention(out_channel)

    def forward(self, x):
        x = self.deconv(x)
        output = x * self.ca(x)
        return output


if __name__ == '__main__':
    pass

