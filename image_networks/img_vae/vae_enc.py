# imports
import torch.nn as nn

"""
VAE Encoder Module VAE

"""


class VAEEncoder(nn.Module):

    def __init__(self, bnd):
        super(VAEEncoder, self).__init__()

        self.encoder = nn.Sequential(

            # H x W -> H/2 X H/2
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
                dilation=1,
                bias=True
            ),

            nn.ReLU(),

            # H/2 x W/2 -> H/4 x W/4
            nn.Conv2d(
                in_channels=64,
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=1,
                dilation=1,
                bias=True
            ),

            nn.ReLU(),

            # H/4 x W/4 -> H/8 x W/8
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                stride=2,
                padding=1,
                dilation=1,
                bias=True
            ),

            nn.ReLU(),

            # H/8 x W/8 -> H/16 x W/16
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=2,
                padding=1,
                dilation=1,
                bias=True
            ),

            nn.ReLU()
        )

        # mean
        self.conv_mu = nn.Conv2d(
            in_channels=512,
            out_channels=bnd,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=True
        )

        # log variance
        self.conv_logvar = nn.Conv2d(
            in_channels=512,
            out_channels=bnd,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=True
        )

    def forward(self, x):
        # encode x
        x = self.encoder(x)
        mu, logvar = self.conv_mu(x), self.conv_logvar(x)
        return mu, logvar
