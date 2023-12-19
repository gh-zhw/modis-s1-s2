import torch
import torch.nn as nn
from torchsummary import summary
from net_block import ConvCBAMBlock, DeconvCBAMBlock, ConvBlock, UpsampleCBAMBlock


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv_cbam_block_1 = ConvCBAMBlock(18, 24, kernel_size=3, stride=2, padding=1)
        self.conv_cbam_block_2 = ConvCBAMBlock(24, 32, kernel_size=3, stride=2, padding=2)
        self.conv_cbam_block_3 = ConvCBAMBlock(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv_cbam_block_4 = ConvCBAMBlock(64, 128, kernel_size=2, stride=2, padding=0)
        self.conv_cbam_block_5 = ConvCBAMBlock(128, 256, kernel_size=2, stride=2, padding=0)
        self.conv_cbam_block_6 = ConvCBAMBlock(256, 512, kernel_size=2, stride=2, padding=1)

        self.deconv_cbam_block_1 = DeconvCBAMBlock(518, 256, kernel_size=4, stride=1, padding=0)
        self.deconv_cbam_block_2 = DeconvCBAMBlock(512, 256, kernel_size=2, stride=2, padding=0)
        self.deconv_cbam_block_3 = DeconvCBAMBlock(384, 192, kernel_size=2, stride=2, padding=0)
        self.deconv_cbam_block_4 = DeconvCBAMBlock(256, 128, kernel_size=2, stride=2, padding=0)
        self.deconv_cbam_block_5 = DeconvCBAMBlock(160, 64, kernel_size=3, stride=2, padding=2)
        self.deconv_cbam_block_6 = DeconvCBAMBlock(88, 32, kernel_size=2, stride=2, padding=0)

        self.res_block_1 = nn.Conv2d(18, 24, kernel_size=3, stride=2, padding=1)
        self.res_block_2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.res_block_3 = nn.Conv2d(128, 256, kernel_size=2, stride=2, padding=0)
        self.res_block_4 = nn.ConvTranspose2d(518, 256, kernel_size=4, stride=1, padding=0)
        self.res_block_5 = nn.ConvTranspose2d(384, 192, kernel_size=2, stride=2, padding=0)
        self.res_block_6 = nn.ConvTranspose2d(160, 64, kernel_size=3, stride=2, padding=2)

        self.conv_block_1 = nn.Sequential(
            ConvBlock(in_channel=6, out_channel=6, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channel=6, out_channel=6, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channel=6, out_channel=6, kernel_size=3, stride=1, padding=1)
        )

        self.conv_block_2 = nn.Sequential(
            ConvCBAMBlock(in_channel=32, out_channel=16, kernel_size=3, stride=1, padding=1),
            ConvCBAMBlock(in_channel=16, out_channel=16, kernel_size=3, stride=1, padding=1),
            ConvCBAMBlock(in_channel=16, out_channel=8, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.tanh = nn.Tanh()

    def forward(self, MODIS_input, S1_input, before_input, after_input):
        # (18, 250, 250)
        S1_ref_size_125 = self.conv_cbam_block_1(torch.cat((S1_input, before_input, after_input), dim=1))
        # (24, 125, 125)
        S1_ref_size_64 = self.conv_cbam_block_2(S1_ref_size_125 + self.res_block_1(torch.cat((S1_input, before_input, after_input), dim=1)))
        # (32, 64, 64)
        S1_ref_size_32 = self.conv_cbam_block_3(S1_ref_size_64)
        # (64, 32, 32)
        S1_ref_size_16 = self.conv_cbam_block_4(S1_ref_size_32 + self.res_block_2(S1_ref_size_64))
        # (128, 16, 16)
        S1_ref_size_8 = self.conv_cbam_block_5(S1_ref_size_16)
        # (256, 8, 8)
        S1_ref_size_5 = self.conv_cbam_block_6(S1_ref_size_8 + self.res_block_3(S1_ref_size_16))
        # (512, 5, 5)

        # (6, 5, 5)
        MODIS_size_5 = self.conv_block_1(MODIS_input)
        MODIS_size_8 = self.deconv_cbam_block_1(torch.cat((MODIS_size_5, S1_ref_size_5), dim=1))
        MODIS_size_8 = MODIS_size_8 + self.res_block_4(torch.cat((MODIS_size_5, S1_ref_size_5), dim=1))
        # (256, 8, 8)
        MODIS_size_16 = self.deconv_cbam_block_2(torch.cat((MODIS_size_8, S1_ref_size_8), dim=1))
        # (384, 16, 16)
        MODIS_size_32 = self.deconv_cbam_block_3(torch.cat((MODIS_size_16, S1_ref_size_16), dim=1))
        MODIS_size_32 = MODIS_size_32 + self.res_block_5(torch.cat((MODIS_size_16, S1_ref_size_16), dim=1))
        # (192, 32, 32)
        MODIS_size_64 = self.deconv_cbam_block_4(torch.cat((MODIS_size_32, S1_ref_size_32), dim=1))
        # (128, 64, 64)
        MODIS_size_125 = self.deconv_cbam_block_5(torch.cat((MODIS_size_64, S1_ref_size_64), dim=1))
        MODIS_size_125 = MODIS_size_125 + self.res_block_6(torch.cat((MODIS_size_64, S1_ref_size_64), dim=1))
        # (64, 125, 125)
        MODIS_size_250 = self.deconv_cbam_block_6(torch.cat((MODIS_size_125, S1_ref_size_125), dim=1))
        # (32, 250, 250)

        S2_output = self.conv_block_2(MODIS_size_250)
        # (8, 250, 250)

        S2_output = self.tanh(S2_output)

        return S2_output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm([16, 125, 125]),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=2),
            nn.LayerNorm([32, 64, 64]),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm([16, 32, 32]),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(16, 8, kernel_size=2, stride=2, padding=0),
            nn.LayerNorm([8, 16, 16]),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(8, 4, kernel_size=2, stride=2, padding=0),
            nn.LayerNorm([4, 8, 8]),
            nn.LeakyReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(256, 1),
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
    summary(g.cuda(), [(6, 5, 5), (2, 250, 250), (8, 250, 250), (8, 250, 250)], 8)
    summary(d.cuda(), (8, 250, 250), 8)

