import torch
import torch.nn as nn
from torchsummary import summary
from net_block import ConvCBAMBlock, DeconvCBAMBlock, ConvBlock, UpsampleCBAMBlock


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv_cbam_block_1 = ConvCBAMBlock(2, 4, kernel_size=3, stride=2, padding=1)
        self.conv_cbam_block_2 = ConvCBAMBlock(4, 8, kernel_size=3, stride=2, padding=2)
        self.conv_cbam_block_3 = ConvCBAMBlock(8, 16, kernel_size=3, stride=2, padding=1)
        self.conv_cbam_block_4 = ConvCBAMBlock(16, 32, kernel_size=2, stride=2, padding=0)
        self.conv_cbam_block_5 = ConvCBAMBlock(32, 64, kernel_size=2, stride=2, padding=0)
        self.conv_cbam_block_6 = ConvCBAMBlock(64, 128, kernel_size=2, stride=2, padding=1)

        self.deconv_cbam_block_1 = DeconvCBAMBlock(134, 128, kernel_size=4, stride=1, padding=0)
        self.deconv_cbam_block_2 = DeconvCBAMBlock(192, 96, kernel_size=2, stride=2, padding=0)
        self.deconv_cbam_block_3 = DeconvCBAMBlock(128, 96, kernel_size=2, stride=2, padding=0)
        self.deconv_cbam_block_4 = DeconvCBAMBlock(112, 64, kernel_size=2, stride=2, padding=0)
        self.deconv_cbam_block_5 = DeconvCBAMBlock(72, 48, kernel_size=3, stride=2, padding=2)
        self.deconv_cbam_block_6 = DeconvCBAMBlock(52, 32, kernel_size=2, stride=2, padding=0)

        # self.upsample_cbam_block_1 = UpsampleCBAMBlock(134, 128, out_size=8)
        # self.upsample_cbam_block_2 = UpsampleCBAMBlock(192, 96, out_size=16)
        # self.upsample_cbam_block_3 = UpsampleCBAMBlock(128, 96, out_size=32)
        # self.upsample_cbam_block_4 = UpsampleCBAMBlock(112, 64, out_size=64)
        # self.upsample_cbam_block_5 = UpsampleCBAMBlock(72, 48, out_size=125)
        # self.upsample_cbam_block_6 = UpsampleCBAMBlock(52, 32, out_size=250)

        self.res_block_1 = nn.Conv2d(2, 4, kernel_size=3, stride=2, padding=1)
        self.res_block_2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.res_block_3 = nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=0)
        self.res_block_4 = nn.ConvTranspose2d(128, 96, kernel_size=2, stride=2, padding=0)
        self.res_block_5 = nn.ConvTranspose2d(96, 64, kernel_size=2, stride=2, padding=0)
        self.res_block_6 = nn.ConvTranspose2d(48, 32, kernel_size=2, stride=2, padding=0)

        self.conv_block_1 = nn.Sequential(
            ConvBlock(in_channel=6, out_channel=6, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channel=6, out_channel=6, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channel=6, out_channel=6, kernel_size=3, stride=1, padding=1)
        )

        self.conv_block_2 = nn.Sequential(
            ConvCBAMBlock(in_channel=32, out_channel=16, kernel_size=3, stride=1, padding=1),
            ConvCBAMBlock(in_channel=16, out_channel=16, kernel_size=3, stride=1, padding=1),
            ConvCBAMBlock(in_channel=16, out_channel=8, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channel=8, out_channel=8, kernel_size=1, stride=1, padding=0, is_norm=False),
            ConvBlock(in_channel=8, out_channel=8, kernel_size=1, stride=1, padding=0, is_norm=False),
            ConvBlock(in_channel=8, out_channel=8, kernel_size=1, stride=1, padding=0, is_norm=False),
        )

        self.tanh = nn.Tanh()

    def forward(self, MODIS_input, S1_input):
        # (2, 250, 250)
        S1_size_125 = self.conv_cbam_block_1(S1_input)
        # (4, 125, 125)
        S1_size_64 = self.conv_cbam_block_2(S1_size_125 + self.res_block_1(S1_input))
        # (8, 64, 64)
        S1_size_32 = self.conv_cbam_block_3(S1_size_64)
        # (16, 32, 32)
        S1_size_16 = self.conv_cbam_block_4(S1_size_32 + self.res_block_2(S1_size_64))
        # (32, 16, 16)
        S1_size_8 = self.conv_cbam_block_5(S1_size_16)
        # (64, 8, 8)
        S1_size_5 = self.conv_cbam_block_6(S1_size_8 + self.res_block_3(S1_size_16))
        # (128, 5, 5)

        # (6, 5, 5)
        MODIS_size_5 = self.conv_block_1(MODIS_input)
        MODIS_size_8 = self.deconv_cbam_block_1(torch.cat((MODIS_size_5, S1_size_5), dim=1))
        # (128, 8, 8)
        MODIS_size_16 = self.deconv_cbam_block_2(torch.cat((MODIS_size_8, S1_size_8), dim=1))
        # (96, 16, 16)
        MODIS_size_16 = MODIS_size_16 + self.res_block_4(MODIS_size_8)
        MODIS_size_32 = self.deconv_cbam_block_3(torch.cat((MODIS_size_16, S1_size_16), dim=1))
        # (96, 32, 32)
        MODIS_size_64 = self.deconv_cbam_block_4(torch.cat((MODIS_size_32, S1_size_32), dim=1))
        # (64, 64, 64)
        MODIS_size_64 = MODIS_size_64 + self.res_block_5(MODIS_size_32)
        MODIS_size_125 = self.deconv_cbam_block_5(torch.cat((MODIS_size_64, S1_size_64), dim=1))
        # (48, 125, 125)
        MODIS_size_250 = self.deconv_cbam_block_6(torch.cat((MODIS_size_125, S1_size_125), dim=1))
        MODIS_size_250 = MODIS_size_250 + self.res_block_6(MODIS_size_125)
        # (32, 250, 250)

        # # (6, 5, 5)
        # MODIS_size_5 = self.conv_block_1(MODIS_input)
        # MODIS_size_8 = self.upsample_cbam_block_1(torch.cat((MODIS_size_5, S1_size_5), dim=1))
        # # (128, 8, 8)
        # MODIS_size_16 = self.upsample_cbam_block_2(torch.cat((MODIS_size_8, S1_size_8), dim=1))
        # # (96, 16, 16)
        # MODIS_size_16 = MODIS_size_16 + self.res_block_4(MODIS_size_8)
        # MODIS_size_32 = self.upsample_cbam_block_3(torch.cat((MODIS_size_16, S1_size_16), dim=1))
        # # (96, 32, 32)
        # MODIS_size_64 = self.upsample_cbam_block_4(torch.cat((MODIS_size_32, S1_size_32), dim=1))
        # # (64, 64, 64)
        # MODIS_size_64 = MODIS_size_64 + self.res_block_5(MODIS_size_32)
        # MODIS_size_125 = self.upsample_cbam_block_5(torch.cat((MODIS_size_64, S1_size_64), dim=1))
        # # (48, 125, 125)
        # MODIS_size_250 = self.upsample_cbam_block_6(torch.cat((MODIS_size_125, S1_size_125), dim=1))
        # MODIS_size_250 = MODIS_size_250 + self.res_block_6(MODIS_size_125)
        # # (32, 250, 250)

        S2_output = self.conv_block_2(MODIS_size_250)
        # (8, 250, 250)

        S2_output = self.tanh(S2_output)

        return S2_output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=2),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=2, stride=2, padding=0),
            nn.LayerNorm(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, kernel_size=2, stride=2, padding=0),
            nn.LayerNorm(8),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(512, 1),
            # nn.Sigmoid()
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
    summary(g.cuda(), [(6, 5, 5), (2, 250, 250)], 32)
    # summary(d.cuda(), (16, 250, 250), 8)

