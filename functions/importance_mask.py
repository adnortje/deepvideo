# imports
import math
import torch
from torch.autograd import Function

"""
QuantisedImportanceMask
    
    creates a mask which zeros out unnecessary bits for video reconstruction
    
    Args:
        p (torch.tensor) : soft importance map (B, 1, T, H, W)
        n (int)          : encoder output channels or bottleneck-depth
        L (int)          : importance levels, such that number bits per level = n // L
    
    Ref:
        https://arxiv.org/abs/1703.10553
"""


class QuantisedImportanceMask(Function):

    @staticmethod
    def forward(ctx, p, L, n):

        # extract dim
        b, _, t, h, w = p.size()

        # def zero mask
        mask = torch.zeros(b, n, t, h, w).to(p.device)

        # number bits per importance level
        if n % L != 0:
            raise ValueError("n % L must = 0!")

        n_bits = n / L

        # save for backward computation
        ctx.n = n
        ctx.L = L
        ctx.save_for_backward(p)

        # quantise importance map
        q_map = torch.floor(L * p) * n_bits
        q_map = q_map.squeeze(dim=1)

        # (B, C, T, H, W) -> (C, B, T, H, W)
        mask = mask.permute(1, 0, 2, 3, 4)

        # populate importance mask
        for k in range(mask.size(0)):
            mask[k, (k <= q_map)] = 1.0

        # (C, B, T, H, W) -> (B, C, T, H, W)
        mask = mask.permute(1, 0, 2, 3, 4)

        return mask

    @staticmethod
    def backward(ctx, grad_output):

        # context
        n = ctx.n
        L = ctx.L
        p, = ctx.saved_tensors

        # (B, C, T, H, W) -> (B, T, H, W)
        p = p.squeeze(dim=1)

        # def dm/dp
        d_m = torch.zeros_like(grad_output)

        # (B, C, T, H, W) -> (C, B, T, H, W)
        d_m = d_m.permute(1, 0, 2, 3, 4)

        # populate gradient
        for k in range(d_m.size(0)):
            d_m[k, ((L*p-1 <= math.ceil(k*L/n))*(math.ceil(k*L/n) < L*p + 2))] = L

        # (C, B, T, H, W) -> (B, C, T, H, W)
        d_m = d_m.permute(1, 0, 2, 3, 4)

        # Chain Rule
        grad_input = d_m * grad_output.clone()

        return grad_input, None, None

