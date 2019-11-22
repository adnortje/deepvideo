# import
import torch
import torch.nn as nn
from .functional import Backward
import torch.nn.functional as f


class Regularization(nn.Module):

    def __init__(self, intLevel):
        super(Regularization, self).__init__()

        self.dblBackward = [0.0, 0.0, 10.0, 5.0, 2.5, 1.25, 0.625][intLevel]

        self.intUnfold = [0, 0, 7, 5, 5, 3, 3][intLevel]

        if intLevel >= 5:
            self.moduleFeat = nn.Sequential()

        elif intLevel < 5:
            self.moduleFeat = nn.Sequential(

                nn.Conv2d(
                    in_channels=[0, 0, 32, 64, 96, 128, 192][intLevel],
                    out_channels=128,
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
                in_channels=[0, 0, 131, 131, 131, 131, 195][intLevel],
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
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.LeakyReLU(
                inplace=False,
                negative_slope=0.1
            )
        )

        if intLevel >= 5:
            self.moduleDist = nn.Sequential(
                nn.Conv2d(
                    in_channels=32,
                    out_channels=[0, 0, 49, 25, 25, 9, 9][intLevel],
                    kernel_size=[0, 0, 7, 5, 5, 3, 3][intLevel],
                    stride=1,
                    padding=[0, 0, 3, 2, 2, 1, 1][intLevel])
            )

        elif intLevel < 5:
            self.moduleDist = nn.Sequential(

                nn.Conv2d(
                    in_channels=32,
                    out_channels=[0, 0, 49, 25, 25, 9, 9][intLevel],
                    kernel_size=([0, 0, 7, 5, 5, 3, 3][intLevel], 1),
                    stride=1,
                    padding=([0, 0, 3, 2, 2, 1, 1][intLevel], 0)
                ),

                nn.Conv2d(
                    in_channels=[0, 0, 49, 25, 25, 9, 9][intLevel],
                    out_channels=[0, 0, 49, 25, 25, 9, 9][intLevel],
                    kernel_size=(1, [0, 0, 7, 5, 5, 3, 3][intLevel]),
                    stride=1,
                    padding=(0, [0, 0, 3, 2, 2, 1, 1][intLevel])
                )
            )

        self.moduleScaleX = nn.Conv2d(
            in_channels=[0, 0, 49, 25, 25, 9, 9][intLevel],
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.moduleScaleY = nn.Conv2d(
            in_channels=[0, 0, 49, 25, 25, 9, 9][intLevel],
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, tensorFirst, tensorSecond, tensorFeaturesFirst, tensorFeaturesSecond, tensorFlow):
        tensorDifference = (
                tensorFirst - Backward(
            tensorInput=tensorSecond,
            tensorFlow=tensorFlow * self.dblBackward)
        ).pow(2.0).sum(1, True).sqrt()

        tensorDist = self.moduleDist(
            self.moduleMain(
                torch.cat(
                    (
                        tensorDifference,
                        tensorFlow - tensorFlow.view(tensorFlow.size(0), 2, -1).mean(
                            2, True
                        ).view(tensorFlow.size(0), 2, 1, 1),
                        self.moduleFeat(tensorFeaturesFirst)
                    ),
                    1
                )
            )
        )

        tensorDist = tensorDist.pow(2.0).neg()
        tensorDist = (tensorDist - tensorDist.max(1, True)[0]).exp()

        tensorDivisor = tensorDist.sum(1, True).reciprocal()

        tensorScaleX = self.moduleScaleX(
            tensorDist * f.unfold(
                input=tensorFlow[:, 0:1, :, :],
                kernel_size=self.intUnfold,
                stride=1,
                padding=int((self.intUnfold - 1) / 2)
            ).view_as(tensorDist)) * tensorDivisor

        tensorScaleY = self.moduleScaleY(
            tensorDist * f.unfold(
                input=tensorFlow[:, 1:2, :, :],
                kernel_size=self.intUnfold,
                stride=1,
                padding=int((self.intUnfold - 1) / 2)
            ).view_as(tensorDist)) * tensorDivisor

        return torch.cat([tensorScaleX, tensorScaleY], 1)
