# import
import torch.nn as nn


"""
I-Frame Feature Extractor

"""


class IFrameFeatExtractor(nn.Module):

    def __init__(self):
        super(IFrameFeatExtractor, self).__init__()

        self.conv_64 = nn.Sequential(

            # H x W
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True
            ),

            nn.ReLU()
        )

        self.conv_128 = nn.Sequential(

            # H x W-> H/2 x W/2
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True
            ),

            nn.ReLU()
        )

        self.conv_256 = nn.Sequential(

            # H/2 x W/2 -> H/4 x W/4
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True
            ),

            nn.ReLU()
        )

        self.conv_512 = nn.Sequential(

            # H/4 x W/4 -> H/8 x W/8
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True
            ),

            nn.ReLU()
        )

    def forward(self, x):

        # (B, C, D, H, W) -> (B, C, H, W)
        x = x.squeeze(2)

        # extract I-frame features
        x1 = self.conv_64(x)
        x2 = self.conv_128(x1)
        x3 = self.conv_256(x2)
        x4 = self.conv_512(x3)

        # dictionary of I-Frame features
        x = {"512": x4, "256": x3, "128": x2, "64": x1}

        return x
