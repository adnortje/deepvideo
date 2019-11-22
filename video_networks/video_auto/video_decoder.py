# imports
import torch
import torch.nn as nn
from modules import PixelShuffle3D


"""
Video Decoder Module

    used to decode video frames
"""


class VideoDecoder(nn.Module):

    def __init__(self, bnd, stateful=False):
        super(VideoDecoder, self).__init__()

        # stateful decoder
        self.stateful = stateful

        if stateful:
            k = 2
        else:
            k = 1

        # decoder components
        self.dec_in = nn.Sequential(

            # D/8 x H/8 x W/8
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

        self.dec_512 = nn.Sequential(
            # D/8 x H/8 x W/8
            nn.Conv3d(
                in_channels=512*k,
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

        self.dec_256 = nn.Sequential(
            # D/4 x H/4 x W/4
            nn.Conv3d(
                in_channels=256*k,
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

        self.dec_128 = nn.Sequential(
            # D/2 x H/2 x W/2
            nn.Conv3d(
                in_channels=128*k,
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

        self.dec_64 = nn.Sequential(
            # D x H x W
            nn.Conv3d(
                in_channels=64*k,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                bias=True
            ),

            nn.ReLU()
        )

        self.dec_out = nn.Sequential(
            # D x H x W
            nn.Conv3d(
                in_channels=32,
                out_channels=3,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                bias=True
            ),

            nn.Tanh()
        )

    def forward(self, x, dec_state=None):
        # decode with state features
        x = self.dec_in(x)

        x = self.dec_512(
            self.concat_state(x, dec_state)
        )

        x = self.dec_256(
            self.concat_state(x, dec_state)
        )

        x = self.dec_128(
            self.concat_state(x, dec_state)
        )

        x = self.dec_64(
            self.concat_state(x, dec_state)
        )

        x = self.dec_out(x)

        return x

    def concat_state(self, x, dec_state):

        if not self.stateful:
            # no state
            return x

        # concatenate x with state information
        x = torch.cat(
            (x, dec_state[str(x.size(1))]),
            dim=1
        )
        return x

