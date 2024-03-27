import torch
import torch.nn as nn
from torchsummary import summary
from net_block import ConvCBAMBlock, DeconvCBAMBlock, ConvBlock


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv_cbam_block_1 = ConvCBAMBlock(18, 36, kernel_size=3, stride=2, padding=1, norm="BN")
        self.conv_cbam_block_2 = ConvCBAMBlock(36, 72, kernel_size=3, stride=2, padding=2, norm="BN")
        self.conv_cbam_block_3 = ConvCBAMBlock(72, 144, kernel_size=3, stride=2, padding=1, norm="BN")
        self.conv_cbam_block_4 = ConvCBAMBlock(144, 288, kernel_size=2, stride=2, padding=0, norm="BN")
        self.conv_cbam_block_5 = ConvCBAMBlock(288, 576, kernel_size=2, stride=2, padding=0, norm="BN")
        self.conv_cbam_block_6 = ConvCBAMBlock(576, 1152, kernel_size=2, stride=2, padding=1, norm="BN")

        self.deconv_cbam_block_1 = DeconvCBAMBlock(1158, 579, kernel_size=4, stride=1, padding=0, norm="BN")
        self.deconv_cbam_block_2 = DeconvCBAMBlock(1155, 578, kernel_size=2, stride=2, padding=0, norm="BN")
        self.deconv_cbam_block_3 = DeconvCBAMBlock(866, 433, kernel_size=2, stride=2, padding=0, norm="BN")
        self.deconv_cbam_block_4 = DeconvCBAMBlock(577, 288, kernel_size=2, stride=2, padding=0, norm="BN")
        self.deconv_cbam_block_5 = DeconvCBAMBlock(360, 90, kernel_size=3, stride=2, padding=2, norm="BN")
        self.deconv_cbam_block_6 = DeconvCBAMBlock(126, 32, kernel_size=2, stride=2, padding=0, norm="BN")

        self.conv_block_1 = ConvCBAMBlock(6, 6, kernel_size=3, stride=1, padding=1, norm="BN")

        self.conv_block_2 = nn.Sequential(
            ConvCBAMBlock(in_channel=50, out_channel=24, kernel_size=3, stride=1, padding=1, norm="BN"),
            ConvCBAMBlock(in_channel=24, out_channel=16, kernel_size=3, stride=1, padding=1, norm="BN"),
            ConvCBAMBlock(in_channel=16, out_channel=8, kernel_size=3, stride=1, padding=1, norm="BN"),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.tanh = nn.Tanh()

    def forward(self, MODIS_input, S1_input, before_input, after_input):
        combined_input = torch.cat((S1_input, before_input, after_input), dim=1)
        # (18, 250, 250)
        S1_ref_size_125 = self.conv_cbam_block_1(combined_input)
        # (36, 125, 125)
        S1_ref_size_64 = self.conv_cbam_block_2(S1_ref_size_125)
        # (72, 64, 64)
        S1_ref_size_32 = self.conv_cbam_block_3(S1_ref_size_64)
        # (144, 32, 32)
        S1_ref_size_16 = self.conv_cbam_block_4(S1_ref_size_32)
        # (288, 16, 16)
        S1_ref_size_8 = self.conv_cbam_block_5(S1_ref_size_16)
        # (576, 8, 8)
        S1_ref_size_4 = self.conv_cbam_block_6(S1_ref_size_8)
        # (1152, 4, 4)

        MODIS_size_4 = self.conv_block_1(MODIS_input)
        # (6, 4, 4)
        MODIS_size_8 = self.deconv_cbam_block_1(torch.cat((MODIS_size_4, S1_ref_size_4), dim=1))
        # (579, 8, 8)
        MODIS_size_16 = self.deconv_cbam_block_2(torch.cat((MODIS_size_8, S1_ref_size_8), dim=1))
        # (578, 16, 16)
        MODIS_size_32 = self.deconv_cbam_block_3(torch.cat((MODIS_size_16, S1_ref_size_16), dim=1))
        # (433, 32, 32)
        MODIS_size_64 = self.deconv_cbam_block_4(torch.cat((MODIS_size_32, S1_ref_size_32), dim=1))
        # (288, 64, 64)
        MODIS_size_125 = self.deconv_cbam_block_5(torch.cat((MODIS_size_64, S1_ref_size_64), dim=1))
        # (90, 125, 125)
        MODIS_size_250 = self.deconv_cbam_block_6(torch.cat((MODIS_size_125, S1_ref_size_125), dim=1))
        # (32, 250, 250)

        S2_output = self.conv_block_2(torch.cat((MODIS_size_250, combined_input), dim=1))
        # (50, 250, 250)

        S2_output = self.tanh(S2_output)

        return S2_output


class Discriminator(nn.Module):
    def __init__(self, output_sig=True):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(20),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(20, 40, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(40),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(40, 80, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(80),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(80, 40, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(40),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(40, 20, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(20),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(20, 10, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(10),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(10, 5, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(5),
            nn.LeakyReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(20, 1),
        )

        if output_sig:
            self.model.add_module("sigmoid", nn.Sigmoid())

    def forward(self, _input):
        return self.model(_input)


if __name__ == '__main__':
    g = Generator()
    d = Discriminator()
    # summary(g.cuda(), [(6, 5, 5), (2, 250, 250), (8, 250, 250), (8, 250, 250)], 8)
    summary(d.cuda(), (10, 250, 250), 8)

