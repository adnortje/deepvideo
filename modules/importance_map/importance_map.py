# import
import torch.nn as nn
from functions import QuantisedImportanceMask

"""
ImportanceMap
    
    Returns masked bitmap and soft importance map for Entropy loss
    
    Args:
        input_dim (int) : channel dimension of input feature map
        n         (int) : number of encoder output channels or bottleneck-depth
        L         (int) : importance levels, such that bits per level = n / L
    
    Ref:
        https://arxiv.org/abs/1703.10553
"""


class ImportanceMap(nn.Module):

    def __init__(self, input_dim, n, L):
        super(ImportanceMap, self).__init__()

        # input feature channels
        self.input_dim = input_dim

        # encoder output channels (bottleneck-depth)
        self.n = n

        # importance levels
        self.L = L

        # importance network
        self.imp_net = nn.Sequential(

            nn.Conv3d(
                in_channels=input_dim,
                out_channels=64,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=True
            ),

            nn.ReLU(),

            nn.Conv3d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=True
            ),

            nn.ReLU(),

            nn.Conv3d(
                in_channels=32,
                out_channels=1,
                kernel_size=1,
                padding=0,
                stride=1,
                bias=True
            ),

            nn.Sigmoid()
        )

    def forward(self, x, feat_map):
        # importance map -> (0, 1)
        p = self.imp_net(feat_map)

        # importance mask
        q_mask = QuantisedImportanceMask.apply(p, self.L, self.n)

        # mask bits
        x = q_mask * x

        return x, p

