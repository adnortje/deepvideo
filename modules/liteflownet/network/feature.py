# imports
import torch.nn as nn

"""
Feature Extraction

"""


class Features(nn.Module):

    def __init__(self):
        super(Features, self).__init__()

        self.moduleOne = nn.Sequential(

            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=7,
                stride=1,
                padding=3
            ),

            nn.LeakyReLU(
                negative_slope=0.1
            )
        )

        self.moduleTwo = nn.Sequential(

            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.LeakyReLU(
                negative_slope=0.1
            ),

            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.LeakyReLU(
                negative_slope=0.1
            ),

            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.LeakyReLU(
                negative_slope=0.1
            )
        )

        self.moduleThr = nn.Sequential(

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.LeakyReLU(
                negative_slope=0.1
            ),

            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.LeakyReLU(
                negative_slope=0.1
            )
        )

        self.moduleFou = nn.Sequential(

            nn.Conv2d(
                in_channels=64,
                out_channels=96,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.LeakyReLU(
                negative_slope=0.1
            ),

            nn.Conv2d(
                in_channels=96,
                out_channels=96,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.LeakyReLU(
                negative_slope=0.1
            )
        )

        self.moduleFiv = nn.Sequential(

            nn.Conv2d(
                in_channels=96,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.LeakyReLU(
                negative_slope=0.1
            )
        )

        self.moduleSix = nn.Sequential(

            nn.Conv2d(
                in_channels=128,
                out_channels=192,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.LeakyReLU(
                negative_slope=0.1
            )
        )

    def forward(self, x):

        x1 = self.moduleOne(x)
        x2 = self.moduleTwo(x1)
        x3 = self.moduleThr(x2)
        x4 = self.moduleFou(x3)
        x5 = self.moduleFiv(x4)
        x6 = self.moduleSix(x5)

        return [x1, x2, x3, x4, x5, x6]
