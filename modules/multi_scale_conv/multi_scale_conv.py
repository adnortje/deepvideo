# imports
import math
import torch
import torch.nn as nn

"""
2D Realization of 'Multi-Scale Context Aggregation by Dilated Convolutions'

    ref:
        https://arxiv.org/pdf/1709.08855.pdf
        https://arxiv.org/abs/1511.07122.pdf
"""


class MultiScaleConv2D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel, stride, padding, dilation, bias=True):
        super(MultiScaleConv2D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        if self.out_channels // len(dilation) <= 0:
            # sanity check
            raise ValueError("Specified Number of Output Channels is too few!")

        # def conv parameters
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # list of multi-scale convolutions
        self.multi_conv = nn.ModuleList([
            nn.Sequential(

                # conv layer with dilation factor
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels//len(dilation),
                    kernel_size=kernel[i],
                    stride=stride[i],
                    padding=padding[i],
                    dilation=dilation[i],
                    bias=bias
                ),

                # activation
                nn.ReLU(),

                # batch normalization
                nn.BatchNorm2d(
                    num_features=out_channels // len(dilation),
                )

            ) for i in range(len(dilation))
        ])

    def forward(self, x):

        opts = []

        for conv in self.multi_conv:

            # append outputs
            opts.append(conv(x))

        # stack outputs
        opts = torch.cat(opts, dim=1)

        return opts


"""
2D Realization of 'Multi-Scale Context Aggregation by Dilated Convolutions'

    ref:
        https://arxiv.org/pdf/1709.08855.pdf
        https://arxiv.org/abs/1511.07122.pdf
"""


class MultiScaleConv3D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel, stride, padding, dilation, bias=True):
        super(MultiScaleConv3D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        if self.out_channels // len(dilation) <= 0:
            # sanity check
            raise ValueError("Specified Number of Output Channels is too few!")

        # def conv parameters
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # list of multi-scale convolutions
        self.multi_conv = nn.ModuleList([
            nn.Sequential(

                # conv layer with dilation factor
                nn.Conv3d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels//len(dilation),
                    kernel_size=kernel[i],
                    stride=stride[i],
                    padding=padding[i],
                    dilation=dilation[i],
                    bias=bias
                ),

                # activation
                nn.ReLU(),

                # batch normalization
                nn.BatchNorm3d(
                    num_features=out_channels // len(dilation),
                )

            ) for i in range(len(dilation))
        ])

    def forward(self, x):

        opts = []

        for conv in self.multi_conv:

            # append outputs
            opts.append(conv(x))

        # stack outputs
        opts = torch.cat(opts, dim=1)

        return opts

