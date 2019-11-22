# imports
import torch
import torch.nn as nn
from torch.nn.functional import _pair

"""
RateLoss
    
    Penalizes system for using excessive number of bits
    
    Args:
        r (int)      : desired or threshold bitrate in bpp
        beta (float) : weighting term (from paper [0.0001, 0.2])
        
    Ref:
        https://arxiv.org/abs/1703.10553
    
"""


class RateLoss(nn.Module):

    def __init__(self, beta, r0, bnd, n_gop=16, f_s=(64, 64)):
        super(RateLoss, self).__init__()

        # compression factor
        self.beta = beta

        # desired bpp
        self.r0 = r0

        # frame size, GOP number & bottleneck-depth
        self.bnd = bnd
        self.f_s = _pair(f_s)
        self.n_gop = n_gop

    def forward(self, x):

        # target bitrate
        h, w = self.f_s
        r = self.r0 * self.n_gop * h * w / self.bnd

        # no. bits designated by importance map
        x = torch.sum(x) / x.size(0)

        # only penalize for bitrates over r
        if x > r:
            loss = self.beta * (x - r)
        else:
            loss = 0

        return loss
