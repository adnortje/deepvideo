# imports
import torch
import torch.nn as nn
from modules import PixelShuffle3D


"""
B-Frame Video Decoder Module

"""


class BFrameDecoder(nn.Module):

    def __init__(self, bnd):
        super(BFrameDecoder, self).__init__()

        # decoder components
        self.dec_in = nn.Sequential(

            # D/8 x H/16 x W/16
            nn.Conv3d(
                in_channels=bnd,
                out_channels=512,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                bias=True
            ),

            nn.ReLU()

        )

        self.dec_1024 = nn.Sequential(

            # D/8 x H/8 x W/8
            nn.Conv3d(
                in_channels=1024,
                out_channels=2048,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                bias=True
            ),

            nn.ReLU(),

            # Depth-to-Space
            PixelShuffle3D(
                upscale_factor=2
            )

        )

        self.dec_512 = nn.Sequential(

            # D/4 x H/4 x W/4
            nn.Conv3d(
                in_channels=512,
                out_channels=1024,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                bias=True
            ),

            nn.ReLU(),

            # Depth-to-Space
            PixelShuffle3D(
                upscale_factor=2
            )
        )

        self.dec_256 = nn.Sequential(

            # D/2 x H/2 x W/2
            nn.Conv3d(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                bias=True
            ),

            nn.ReLU(),

            # Depth-to-Space
            PixelShuffle3D(
                upscale_factor=2
            )
        )

        self.dec_out = nn.Sequential(

            # D x H x W
            nn.Conv3d(
                in_channels=128,
                out_channels=3,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                bias=True
            ),

            nn.Tanh()
        )

    def forward(self, x, i_feat):
        # decode with I-Frame features

        x = self.dec_in(x)

        x = self.concat_feat(x, i_feat["512"])

        x = self.dec_1024(x)

        x = self.concat_feat(x, i_feat["256"])

        x = self.dec_512(x)

        x = self.concat_feat(x, i_feat["128"])

        x = self.dec_256(x)

        x = self.concat_feat(x, i_feat["64"])

        x = self.dec_out(x)

        return x

    @staticmethod
    def concat_feat(v, feat):
        v = torch.cat(
            (v, feat.expand(-1, -1, v.size(2), -1, -1)),
            dim=1
        )
        return v
