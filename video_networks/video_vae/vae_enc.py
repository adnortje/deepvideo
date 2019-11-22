# imports
import torch
import torch.nn as nn
from .m_cond import MotionCond
from modules import MultiScaleConv2D
from .ms_video_enc import MultiScaleVideoEncoder

"""
Video VAE Encoder Module

"""


class VideoVAEEncoder(nn.Module):

    def __init__(self, bnd, vae_bnd=128):
        super(VideoVAEEncoder, self).__init__()

        # bottleneck depth
        self.bnd = bnd

        # Motion Encoder
        self.motion_encoder = MultiScaleVideoEncoder(
            bnd=bnd
        )

        # Motion Conditioning Network
        self.motion_cond = MotionCond(
            bnd=bnd
        )

        # VAE Encoder
        self.vae_enc_in = nn.Sequential(

            # H x W -> H/2 x W/2
            MultiScaleConv2D(
                in_channels=3,
                out_channels=64,
                kernel=[3, 3, 3, 3],
                stride=[2, 2, 2, 2],
                padding=[1, 2, 4, 8],
                dilation=[1, 2, 4, 8],
                bias=True
            )

        )

        self.vae_enc_128 = nn.Sequential(

            # H/2 x W/2 -> H/4 x W/4
            MultiScaleConv2D(
                in_channels=128,
                out_channels=256,
                kernel=[3, 3, 3, 3],
                stride=[2, 2, 2, 2],
                padding=[1, 2, 4, 8],
                dilation=[1, 2, 4, 8],
                bias=True
            )

        )

        self.vae_enc_512 = nn.Sequential(

            # H/4 x W/4 -> H/8 x W/8
            MultiScaleConv2D(
                in_channels=512,
                out_channels=512,
                kernel=[3, 3, 3, 3],
                stride=[2, 2, 2, 2],
                padding=[1, 2, 4, 8],
                dilation=[1, 2, 4, 8],
                bias=True
            )
        )

        self.vae_enc_out = nn.Sequential(

            # H/8 x W/8 -> H/16 x W/16
            MultiScaleConv2D(
                in_channels=512,
                out_channels=512,
                kernel=[3, 3, 3, 3],
                stride=[2, 2, 2, 2],
                padding=[1, 2, 4, 8],
                dilation=[1, 2, 4, 8],
                bias=True
            )

        )

        # mean
        self.conv_mu = nn.Conv2d(
            in_channels=512,
            out_channels=vae_bnd,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=True
        )

        # log variance
        self.conv_logvar = nn.Conv2d(
            in_channels=512,
            out_channels=vae_bnd,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=True
        )

    def forward(self, x, iframe):

        # motion bits & conditioning features
        b = self.motion_encoder(x)
        m_cond = self.motion_cond(b.squeeze(2))

        # VAE with motion conditioning
        x = self.vae_enc_in(iframe)

        x = self.vae_enc_128(
            torch.cat((x, m_cond['64']), dim=1)
        )

        x = self.vae_enc_512(
            torch.cat((x, m_cond['256']), dim=1)
        )

        x = self.vae_enc_out(x)

        # encode features to means and log-variances
        mu, logvar = self.conv_mu(x), self.conv_logvar(x)

        return mu, logvar
