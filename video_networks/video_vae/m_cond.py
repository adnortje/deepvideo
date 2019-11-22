# imports
import torch.nn as nn


"""
    Motion Conditioning Network
        
        reshapes 3D motion bits to a learned 2D representation

"""


class MotionCond(nn.Module):

    def __init__(self, bnd):
        super(MotionCond, self).__init__()

        # bottleneck depth
        self.bnd = bnd

        self.conv1x1_64 = nn.Sequential(
            # BND x H/8 x W/8 -> 256 x H/2 x W/2

            nn.Conv2d(
                in_channels=bnd,
                out_channels=16*bnd,
                kernel_size=1,
                dilation=1,
                padding=0,
                stride=1,
                bias=True
            ),

            nn.ReLU(),

            nn.PixelShuffle(
                upscale_factor=4
            ),

            nn.Conv2d(
                in_channels=bnd,
                out_channels=64,
                kernel_size=1,
                dilation=1,
                padding=0,
                stride=1,
                bias=True
            ),

            nn.ReLU()
        )

        self.conv1x1_256 = nn.Sequential(
            # BND x H/8 x W/8 -> 256 x H/4 x W/4

            nn.Conv2d(
                in_channels=bnd,
                out_channels=4*bnd,
                kernel_size=1,
                dilation=1,
                padding=0,
                stride=1,
                bias=True
            ),

            nn.ReLU(),

            nn.PixelShuffle(
                upscale_factor=2
            ),

            nn.Conv2d(
                in_channels=bnd,
                out_channels=256,
                kernel_size=1,
                dilation=1,
                padding=0,
                stride=1,
                bias=True
            ),

            nn.ReLU()
        )

    def forward(self, x):
        # reshape motion bits
        x = {"64": self.conv1x1_64(x), "256": self.conv1x1_256(x)}
        return x
