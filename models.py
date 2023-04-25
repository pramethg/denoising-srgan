import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self,
                in_channels,
                out_channels,
                discriminator = False,
                activation = True,
                batch_norm = True,
                **kwargs):
        super(ConvBlock, self).__init__()
        self.activation = activation
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs, bias = not batch_norm)
        self.bnorm = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
        self.act = (
            nn.LeakyReLU(0.2, inplace=True)
            if discriminator
            else nn.PReLU(num_parameters = out_channels)
        )
    def forward(self, x):
        if self.activation:
            return self.act(self.bnorm(self.conv(x)))
        else:
            return self.bnorm(self.conv(x))

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, scale_factor):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * scale_factor ** 2, 3, 1, 1)
        self.redist = nn.PixelShuffle(scale_factor)
        self.activation = nn.PReLU(num_parameters = in_channels)

    def forward(self, x):
        return self.activation(self.redist(self.conv(x)))

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.block1 = ConvBlock(
                in_channels,
                in_channels,
                kernel_size = 3,
                stride = 1,
                padding = 1
            )
        self.block2 = ConvBlock(
                in_channels,
                in_channels,
                kernel_size = 3,
                stride = 1,
                padding = 1,
                activation = False,
            )
    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        return out + x

class Generator(nn.Module):
    def __init__(self, in_channels, num_channels, num_blocks, upsample = False):
        super(Generator, self).__init__()
        self.upsample = upsample
        self.conv1 = ConvBlock(
                        in_channels, 
                        num_channels, 
                        kernel_size = 9, 
                        stride = 1,
                        padding = 4,
                        batch_norm = False
                    )
        self.residual = nn.Sequential(*[ResidualBlock(num_channels) for _ in range(num_blocks)])
        self.conv2 = ConvBlock(
                        num_channels, 
                        num_channels, 
                        kernel_size = 3, 
                        stride = 1, 
                        padding = 1, 
                        activation = False
                    )
        self.upscale = nn.Sequential(
                        UpsampleBlock(num_channels, 2), 
                        UpsampleBlock(num_channels, 2)
                    )
        self.conv3 = nn.Conv2d(num_channels, in_channels, kernel_size = 9, stride = 1, padding = 4)

    def forward(self, x):
        out1 = self.conv1(x)
        x = self.residual(out1)
        x = self.conv2(x) + out1
        if self.upsample:
            x = self.upscale(x)
        return torch.tanh(self.conv3(x))

class Discriminator(nn.Module):
    def __init__(self, in_channels, channels = [64, 64, 128, 128, 256, 256, 512, 512]):
        super(Discriminator, self).__init__()
        blocks = []
        for index, channel in enumerate(channels):
            blocks.append(
                ConvBlock(
                    in_channels,
                    channel,
                    kernel_size = 3,
                    stride=1 + index % 2,
                    padding=1,
                    discriminator = True,
                    activation = True,
                    batch_norm = False if index == 0 else True,
                )
            )
            in_channels = channel

        self.blocks = nn.Sequential(*blocks)
        self.classification = nn.Sequential(
                        nn.AdaptiveAvgPool2d((6, 6)),
                        nn.Flatten(),
                        nn.Linear(512*6*6, 1024),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Linear(1024, 1),
                    )
    def forward(self, x):
        x = self.blocks(x)
        return self.classification(x)