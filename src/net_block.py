import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, is_norm=True):
        super(ConvBlock, self).__init__()

        self.is_norm = is_norm
        self.conv_1 = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=False)
        self.norm = nn.InstanceNorm2d(out_channel)
        self.activation = nn.LeakyReLU(0.1, inplace=True)
        self.conv_2 = nn.Conv2d(out_channel, out_channel, 1, 1, 0, bias=False)

    def forward(self, x):
        out = self.conv_1(x)
        if self.is_norm:
            out = self.norm(out)
        out = self.activation(out)
        out = self.conv_2(out)
        return out


class DeconvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, is_norm=True):
        super(DeconvBlock, self).__init__()

        self.is_norm = is_norm
        self.deconv = nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride, padding, bias=False)
        self.norm = nn.InstanceNorm2d(out_channel)
        self.activation = nn.LeakyReLU(0.1, inplace=True)
        self.conv = nn.Conv2d(out_channel, out_channel, 1, 1, 0, bias=False)

    def forward(self, x):
        out = self.deconv(x)
        if self.is_norm:
            out = self.norm(out)
        out = self.activation(out)
        output = self.conv(out)
        return out


class UpsampleBlock(nn.Module):
    def __init__(self, in_channel, out_channel, out_size, is_norm=True):
        super(UpsampleBlock, self).__init__()

        self.is_norm = is_norm
        self.upsample = nn.Upsample(size=out_size, mode='nearest')
        self.conv_1 = nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=False)
        self.norm = nn.InstanceNorm2d(out_channel)
        self.activation = nn.LeakyReLU(0.1, inplace=True)
        self.conv_2 = nn.Conv2d(out_channel, out_channel, 1, 1, 0, bias=False)

    def forward(self, x):
        out = self.upsample(x)
        out = self.conv_1(out)
        if self.is_norm:
            out = self.norm(out)
        out = self.activation(out)
        out = self.conv_2(out)
        return out


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
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, is_norm=True):
        super().__init__()
        self.conv = ConvBlock(in_channel, out_channel, kernel_size, stride, padding, is_norm=is_norm)
        self.CBAM = CBAM(out_channel)
        self.activation = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.CBAM(out)
        out = self.activation(out)
        return out


class DeconvCBAMBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, is_norm=True):
        super().__init__()
        self.deconv = DeconvBlock(in_channel, out_channel, kernel_size, stride, padding, is_norm=is_norm)
        self.CBAM = CBAM(out_channel)
        self.activation = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        out = self.deconv(x)
        out = self.CBAM(out)
        out = self.activation(out)
        return out


class UpsampleCBAMBlock(nn.Module):
    def __init__(self, in_channel, out_channel, out_size, is_norm=True):
        super().__init__()
        self.upsample = UpsampleBlock(in_channel, out_channel, out_size, is_norm=is_norm)
        self.CBAM = CBAM(out_channel)
        self.activation = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        out = self.upsample(x)
        out = self.CBAM(out)
        out = self.activation(out)
        return out


if __name__ == '__main__':
    pass
