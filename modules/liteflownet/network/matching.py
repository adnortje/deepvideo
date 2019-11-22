# imports
import torch.nn as nn
import torch.nn.functional as f
from .functional import Backward
from .correlation import FunctionCorrelation


"""
Matching

"""


class Matching(nn.Module):

    def __init__(self, intLevel):
        super(Matching, self).__init__()

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

        if intLevel == 6:
            self.moduleUpflow = None

        elif intLevel != 6:
            self.moduleUpflow = nn.ConvTranspose2d(
                in_channels=2,
                out_channels=2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
                groups=2
            )

        if intLevel >= 4:
            self.moduleUpcorr = None

        elif intLevel < 4:
            self.moduleUpcorr = nn.ConvTranspose2d(
                in_channels=49,
                out_channels=49,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
                groups=49
            )

        self.moduleMain = nn.Sequential(

            nn.Conv2d(
                in_channels=49,
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
                padding=[0, 0, 3, 2, 2, 1, 1][intLevel])
        )

    def forward(self, tensorFirst, tensorSecond, tensorFeaturesFirst, tensorFeaturesSecond, tensorFlow):

        tensorFeaturesFirst = self.moduleFeat(tensorFeaturesFirst)
        tensorFeaturesSecond = self.moduleFeat(tensorFeaturesSecond)

        if tensorFlow is not None:
            tensorFlow = self.moduleUpflow(tensorFlow)

        if tensorFlow is not None:
            tensorFeaturesSecond = Backward(
                tensorInput=tensorFeaturesSecond,
                tensorFlow=tensorFlow * self.dblBackward
            )

        if self.moduleUpcorr is None:
            tensorCorrelation = f.leaky_relu(
                input=FunctionCorrelation(
                    tensorFirst=tensorFeaturesFirst,
                    tensorSecond=tensorFeaturesSecond,
                    intStride=1
                ),
                negative_slope=0.1,
                inplace=False
            )

        elif self.moduleUpcorr is not None:
            tensorCorrelation = self.moduleUpcorr(
                f.leaky_relu(
                    input=FunctionCorrelation(
                        tensorFirst=tensorFeaturesFirst,
                        tensorSecond=tensorFeaturesSecond,
                        intStride=2
                    ),
                    negative_slope=0.1,
                    inplace=False
                )
            )

        return (tensorFlow if tensorFlow is not None else 0.0) + self.moduleMain(tensorCorrelation)
