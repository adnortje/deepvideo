# import
import torch
import torch.nn as nn
from .functional import Backward


"""
Subpixel Class

"""


class Subpixel(nn.Module):

    def __init__(self, intLevel):
        super(Subpixel, self).__init__()

        self.dblBackward = [0.0, 0.0, 10.0, 5.0, 2.5, 1.25, 0.625][intLevel]

        if intLevel != 2:
            self.moduleFeat = nn.Sequential()

        elif intLevel == 2:
            self.moduleFeat = nn.Sequential(
                nn.Conv2d(
                    in_channels=32,
                    out_channels=64,
                    kernel_size=1,
                    stride=1,
                    padding=0
                ),
                nn.LeakyReLU(
                    inplace=False,
                    negative_slope=0.1
                )
            )

        self.moduleMain = nn.Sequential(

            nn.Conv2d(
                in_channels=[0, 0, 130, 130, 194, 258, 386][intLevel],
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.LeakyReLU(
                inplace=False,
                negative_slope=0.1
            ),

            nn.Conv2d(
                in_channels=128,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.LeakyReLU(
                inplace=False,
                negative_slope=0.1
            ),

            nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.LeakyReLU(
                inplace=False,
                negative_slope=0.1
            ),

            nn.Conv2d(
                in_channels=32,
                out_channels=2,
                kernel_size=[0, 0, 7, 5, 5, 3, 3][intLevel],
                stride=1,
                padding=[0, 0, 3, 2, 2, 1, 1][intLevel]
            )
        )

    def forward(self, tensorFirst, tensorSecond, tensorFeaturesFirst, tensorFeaturesSecond, tensorFlow):

        tensorFeaturesFirst = self.moduleFeat(tensorFeaturesFirst)
        tensorFeaturesSecond = self.moduleFeat(tensorFeaturesSecond)

        if tensorFlow is not None:
            tensorFeaturesSecond = Backward(
                tensorInput=tensorFeaturesSecond,
                tensorFlow=tensorFlow * self.dblBackward
            )

        return (tensorFlow if tensorFlow is not None else 0.0) + self.moduleMain(
            torch.cat([tensorFeaturesFirst, tensorFeaturesSecond, tensorFlow], 1))
