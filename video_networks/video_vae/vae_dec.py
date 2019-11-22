# imports
import torch.nn as nn

"""
VAE Decoder Module

"""


class VideoVAEDecoder(nn.Module):

    def __init__(self, bnd):
        super(VideoVAEDecoder, self).__init__()

        self.decoder = nn.Sequential(

            nn.Conv2d(
                in_channels=bnd,
                out_channels=512,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                bias=True
            ),

            nn.ReLU(),

            # H/16 x W/16
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                bias=True
            ),

            nn.ReLU(),

            # Depth-to-Space
            nn.PixelShuffle(
                upscale_factor=2
            ),

            # H/8 x W/8
            nn.Conv2d(
                in_channels=128,
                out_channels=512,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                bias=True
            ),

            nn.ReLU(),

            # Depth-to-Space
            nn.PixelShuffle(
                upscale_factor=2
            ),

            # H/4x W/4
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                bias=True
            ),

            nn.ReLU(),

            # Depth-to-Space
            nn.PixelShuffle(
                upscale_factor=2
            ),

            # H/2 x W/2
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                bias=True
            ),

            nn.ReLU(),

            # Depth-to-Space
            nn.PixelShuffle(
                upscale_factor=2
            ),

            # H x W
            nn.Conv2d(
                in_channels=32,
                out_channels=3,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                bias=True
            ),

            nn.Tanh()
        )

    def forward(self, x):
        x = self.decoder(x)
        return x
