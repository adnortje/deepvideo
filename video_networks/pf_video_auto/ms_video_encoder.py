# import
import torch.nn as nn
from modules import Binarizer, MultiScaleConv3D

"""
Multi-Scale Video Encoder Module

"""


class MultiScaleVideoEncoder(nn.Module):

    def __init__(self, bnd):
        super(MultiScaleVideoEncoder, self).__init__()

        # encoder network
        self.encoder = nn.Sequential(

            # D x H x W
            nn.Conv3d(
                in_channels=3,
                out_channels=64,
                kernel_size=(2, 3, 3),
                stride=1,
                dilation=1,
                padding=(0, 1, 1),
                bias=True
            ),

            nn.ReLU(),

            # D x H x W -> D/2 x H/2 x W/2
            MultiScaleConv3D(
                in_channels=64,
                out_channels=256,
                kernel=[3, 3, 3, 3],
                stride=[2, 2, 2, 2],
                padding=[1, 1, 2, 4],
                dilation=[1, 1, 2, 4],
                bias=True
            ),

            # D/2 x H/2 x W/2 -> D/4 x H/4 x W/4
            MultiScaleConv3D(
                in_channels=256,
                out_channels=512,
                kernel=[3, 3, 3, 3],
                stride=[2, 2, 2, 2],
                padding=[1, 1, 2, 4],
                dilation=[1, 1, 2, 4],
                bias=True
            ),

            # D/4 x H/4 x W/4 -> D/8 x H/8 x W/8
            MultiScaleConv3D(
                in_channels=512,
                out_channels=512,
                kernel=[3, 3, 3, 3],
                stride=[2, 2, 2, 2],
                padding=[1, 1, 2, 4],
                dilation=[1, 1, 2, 4],
                bias=True
            )

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
