import torch
import torch.nn as nn
import torch.nn.functional as F


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    """
    def __init__(self, channels, use_conv):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    """
    def __init__(self, channels, use_conv):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        if use_conv:
            self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
        else:
            self.op = nn.AvgPool2d(2)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


if __name__ == "__main__":
    ipt = torch.randn((2, 16, 64, 64))
    up = Upsample(16, True)
    down = Downsample(16, True)
    print("Input:", ipt.shape)
    print("Upsample:", up(ipt).shape)
    print("Downsample:", down(ipt).shape)
