import torch
import torch.nn as nn
from torchsummary import summary


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(2, 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=3, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.deconv_block_1 = nn.Sequential(
            nn.ConvTranspose2d(70, 64, kernel_size=3, stride=3, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.deconv_block_2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.deconv_block_3 = nn.Sequential(
            nn.ConvTranspose2d(32, 24, kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(24),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.deconv_block_4 = nn.Sequential(
            nn.ConvTranspose2d(24, 16, kernel_size=7, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.deconv_block_5 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input_1, input_2):
        # (4, 125, 125)
        input_1 = self.conv_block_1(input_1)
        # (8, 62, 62)
        input_1 = self.conv_block_2(input_1)
        # (16, 31, 31)
        input_1 = self.conv_block_3(input_1)
        # (32, 15, 15)
        input_1 = self.conv_block_4(input_1)
        # (64, 5, 5)
        input_1 = self.conv_block_5(input_1)

        # 在通道维度合并
        # (64, 15, 15)
        input_2 = self.deconv_block_1(torch.cat((input_1, input_2), dim=1))
        # (32, 30, 30)
        input_2 = self.deconv_block_2(input_2)
        # (24, 60, 60)
        input_2 = self.deconv_block_3(input_2)
        # (16, 125, 125)
        input_2 = self.deconv_block_4(input_2)
        # (8, 250, 250)
        input_2 = self.deconv_block_5(input_2)

        return input_2


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # (8, 250, 250)
            nn.Conv2d(8, 16, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),

            # (16, 125, 125)
            nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # (32, 62, 62)
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # (64, 31, 31)
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # (128, 15, 15)
            nn.Conv2d(128, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2, inplace=True),

            # (1, 5, 5)
            nn.Flatten(),
            nn.Linear(225, 1),
            nn.Sigmoid(),
        )

    def forward(self, _input):
        return self.model(_input)


if __name__ == '__main__':
    g = Generator()
    d = Discriminator()
    # summary(g.cuda(), [(2, 250, 250), (6, 5, 5)], 8)
    summary(d.cuda(), (8, 250, 250), 8)
