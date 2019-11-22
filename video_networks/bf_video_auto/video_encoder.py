# import
import torch.nn as nn
from modules import Binarizer

"""
Video Encoder Module

"""


class VideoEncoder(nn.Module):

    def __init__(self, bnd):
        super(VideoEncoder, self).__init__()

        # encoder network
        self.encoder = nn.Sequential(

            # D x H x W
            nn.Conv3d(
                in_channels=3,
                out_channels=64,
                kernel_size=3,
                stride=1,
                dilation=1,
                padding=(0, 1, 1),
                bias=True
            ),

            nn.ReLU(),

            # D x H x W -> D/2 x H/2 x W/2
            nn.Conv3d(
                in_channels=64,
                out_channels=256,
                kernel_size=3,
                stride=2,
                dilation=1,
                padding=1,
                bias=True
            ),

            nn.ReLU(),

            # D/2 x H/2 x W/2 -> D/4 x H/4 x W/4
            nn.Conv3d(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                stride=2,
                dilation=1,
                padding=1,
                bias=True
            ),

            nn.ReLU(),

            # D/4 x H/4 x W/4 -> D/8 x H/8 x W/8
            nn.Conv3d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=2,
                dilation=1,
                padding=1,
                bias=True
            ),

            nn.ReLU()

        )

        self.binarizer = nn.Sequential(
            # D/8 x H/8 x W/8
            nn.Conv3d(
                in_channels=512,
                out_channels=bnd,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                bias=True
            ),

            nn.Tanh(),

            # binarization layer
            Binarizer()

        )

    def forward(self, x):
        # encoded & binarize video frames
        x = self.encoder(x)
        b = self.binarizer(x)
        return x, b
