import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, bias=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, (in_channel + out_channel) // 2, kernel_size, stride, padding, bias=bias),
            nn.InstanceNorm2d((in_channel + out_channel) // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d((in_channel + out_channel) // 2, out_channel, 1, 1, 0, bias=bias),
            nn.InstanceNorm2d(out_channel),
        )

    def forward(self, x):
        output = self.conv(x)
        return output


class DeconvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, bias=False):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, (in_channel + out_channel) // 2, kernel_size, stride, padding, bias=bias),
            nn.InstanceNorm2d((in_channel + out_channel) // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d((in_channel + out_channel) // 2, out_channel, 1, 1, 0, bias=bias),
            nn.InstanceNorm2d(out_channel),
        )

    def forward(self, x):
        output = self.deconv(x)
        return output


class ChannelAttention(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_MLP(self.avg_pool(x))
        max_out = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv2d = nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channel, ratio=2)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


class ConvCBAMBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, bias=False):
        super().__init__()
        self.conv = ConvBlock(in_channel, out_channel, kernel_size, stride, padding, bias=bias)
        self.CBAM = CBAM(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.CBAM(x)
        output = self.relu(x)
        return output


class DeconvCBAMBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, bias=False):
        super().__init__()
        self.deconv = DeconvBlock(in_channel, out_channel, kernel_size, stride, padding, bias=bias)
        self.CBAM = CBAM(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.deconv(x)
        x = self.CBAM(x)
        output = self.relu(x)
        return output


if __name__ == '__main__':
    pass
