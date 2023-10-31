import torch
import torch.nn as nn
from torchsummary import summary
from net_block import ConvBlock, DeconvBlock, ConvSpatialAttentionBlock, DeconvChannelAttentionBlock


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv_sa_block_1 = ConvSpatialAttentionBlock(2, 4, kernel_size=3, stride=2, padding=1)
        self.conv_sa_block_2 = ConvSpatialAttentionBlock(4, 8, kernel_size=3, stride=2, padding=2)
        self.conv_sa_block_3 = ConvSpatialAttentionBlock(8, 16, kernel_size=3, stride=2, padding=1)
        self.conv_sa_block_4 = ConvSpatialAttentionBlock(16, 32, kernel_size=2, stride=2, padding=0)
        self.conv_sa_block_5 = ConvSpatialAttentionBlock(32, 64, kernel_size=2, stride=2, padding=0)
        self.conv_sa_block_6 = ConvSpatialAttentionBlock(64, 72, kernel_size=2, stride=2, padding=1)

        self.deconv_ca_block_1 = DeconvChannelAttentionBlock(78, 64, kernel_size=4, stride=1, padding=0)
        self.deconv_ca_block_2 = DeconvChannelAttentionBlock(128, 64, kernel_size=2, stride=2, padding=0)
        self.deconv_ca_block_3 = DeconvChannelAttentionBlock(96, 64, kernel_size=2, stride=2, padding=0)
        self.deconv_ca_block_4 = DeconvChannelAttentionBlock(80, 32, kernel_size=2, stride=2, padding=0)
        self.deconv_ca_block_5 = DeconvChannelAttentionBlock(40, 16, kernel_size=3, stride=2, padding=2)
        self.deconv_ca_block_6 = DeconvChannelAttentionBlock(20, 8, kernel_size=2, stride=2, padding=0)

        self.res_block_1 = ConvBlock(in_channel=2, out_channel=4, kernel_size=3, stride=2, padding=1)
        self.res_block_2 = ConvBlock(in_channel=8, out_channel=16, kernel_size=3, stride=2, padding=1)
        self.res_block_3 = ConvBlock(in_channel=32, out_channel=64, kernel_size=2, stride=2, padding=0)
        self.res_block_4 = DeconvBlock(in_channel=64, out_channel=64, kernel_size=2, stride=2, padding=0)
        self.res_block_5 = DeconvBlock(in_channel=64, out_channel=32, kernel_size=2, stride=2, padding=0)
        self.res_block_6 = DeconvBlock(in_channel=16, out_channel=8, kernel_size=2, stride=2, padding=0)

        self.conv_block = nn.Sequential(
            ConvBlock(in_channel=8, out_channel=8, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channel=8, out_channel=8, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channel=8, out_channel=8, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, MODIS_input, S1_input):
        # (2, 250, 250)
        S1_size_125 = self.conv_sa_block_1(S1_input)
        # (4, 125, 125)
        S1_size_64 = self.conv_sa_block_2(S1_size_125 + self.res_block_1(S1_input))
        # (8, 64, 64)
        S1_size_32 = self.conv_sa_block_3(S1_size_64)
        # (16, 32, 32)
        S1_size_16 = self.conv_sa_block_4(S1_size_32 + self.res_block_2(S1_size_64))
        # (32, 16, 16)
        S1_size_8 = self.conv_sa_block_5(S1_size_16)
        # (64, 8, 8)
        S1_size_5 = self.conv_sa_block_6(S1_size_8 + self.res_block_3(S1_size_16))
        # (72, 5, 5)

        # (6, 5, 5)
        a = torch.cat((MODIS_input, S1_size_5), dim=1)
        MODIS_size_8 = self.deconv_ca_block_1(torch.cat((MODIS_input, S1_size_5), dim=1))
        # (64, 8, 8)
        MODIS_size_16 = self.deconv_ca_block_2(torch.cat((MODIS_size_8, S1_size_8), dim=1))
        # (64, 16, 16)
        MODIS_size_16 += self.res_block_4(MODIS_size_8)
        MODIS_size_32 = self.deconv_ca_block_3(torch.cat((MODIS_size_16, S1_size_16), dim=1))
        # (64, 32, 32)
        MODIS_size_64 = self.deconv_ca_block_4(torch.cat((MODIS_size_32, S1_size_32), dim=1))
        # (32, 64, 64)
        MODIS_size_64 += self.res_block_5(MODIS_size_32)
        MODIS_size_125 = self.deconv_ca_block_5(torch.cat((MODIS_size_64, S1_size_64), dim=1))
        # (16, 125, 125)
        MODIS_size_250 = self.deconv_ca_block_6(torch.cat((MODIS_size_125, S1_size_125), dim=1))
        MODIS_size_250 += self.res_block_6(MODIS_size_125)

        S2_output = self.conv_block(MODIS_size_250)

        return S2_output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            ConvBlock(in_channel=8, out_channel=16, kernel_size=3, stride=2, padding=1),
            ConvBlock(in_channel=16, out_channel=32, kernel_size=3, stride=2, padding=2),
            ConvBlock(in_channel=32, out_channel=64, kernel_size=3, stride=2, padding=1),
            ConvBlock(in_channel=64, out_channel=32, kernel_size=2, stride=2, padding=0),
            ConvBlock(in_channel=32, out_channel=16, kernel_size=2, stride=2, padding=0),
            nn.Flatten(),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, _input):
        return self.model(_input)


class GAN(nn.Module):
    def __init__(self, generator, discriminator):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

    def forward(self, MODIS_input, S1_input):
        S2_output = self.generator(MODIS_input, S1_input)
        return S2_output


if __name__ == '__main__':
    g = Generator()
    d = Discriminator()
    # summary(g.cuda(), [(6, 5, 5), (2, 250, 250)], 8)
    summary(d.cuda(), (8, 250, 250), 8)
