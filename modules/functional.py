# import
import torch
import torch.nn.functional as f


"""
Function: pixel_shuffle_3d
    
    Rearranges elements (N, C, D, H, W) -> (N, C/(r^3), D*r, H*r, W*r).
    
    ref:
        https://github.com/pytorch/pytorch/pull/6340/files
"""


def pixel_shuffle_3d(input, upscale_factor):

    # extract dimensions
    batch_size, channels, in_dim, in_height, in_width = input.size()
    channels //= upscale_factor ** 3

    # new dimensions
    out_dim = in_dim * upscale_factor
    out_width = in_width * upscale_factor
    out_height = in_height * upscale_factor

    input_view = input.contiguous().view(
        batch_size, channels, upscale_factor, upscale_factor, upscale_factor, in_dim, in_height, in_width
    )

    shuffle_out = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

    return shuffle_out.view(batch_size, channels, out_dim, out_height, out_width)


"""
Function: pixel_shuffle_2d

    Rearranges elements (N, C, D, H, W) -> (N, C/(r^2), D, H*r, W*r).

    ref:
        https://github.com/pytorch/pytorch/pull/6340/files
"""


def pixel_shuffle_2d(input, upscale_factor):

    # (N, C, D, H, W) -> (N, D, C, H, W)
    input = input.permute(0, 2, 1, 3, 4)

    # extract dimensions
    batch_size, in_dim, channels, in_height, in_width = input.size()
    channels //= upscale_factor ** 2

    # new dimensions
    out_width = in_width * upscale_factor
    out_height = in_height * upscale_factor

    input_view = input.contiguous().view(
        batch_size, in_dim, channels, upscale_factor, upscale_factor, in_height, in_width
    )

    shuffle_out = input_view.permute(0, 1, 2, 5, 3, 6, 4).contiguous()
    shuffle_out = shuffle_out.view(batch_size, in_dim, channels, out_height, out_width)

    return shuffle_out.permute(0, 2, 1, 3, 4)


"""
KLD Loss for Gaussian distribution

    Args:
        mu     (torch.Tensor) : means
        logvar (torch.Tensor) : log variances

    ref:
        https://arxiv.org/abs/1312.6114
        https://github.com/pytorch/examples/blob/master/vae/main.py
"""


def kld_loss_gaussian(mu, logvar, reduction="sum"):

    # KLD Loss
    kld = 1.0 + logvar - mu.pow(2) - logvar.exp()

    if reduction == "sum":
        kld = -0.5 * torch.sum(kld)

    elif reduction == "element_wise_mean":
        kld = -0.5 * torch.mean(kld)

    return kld


"""
End-Point-Error (EPE) Loss for flow vectors

    Args:
        target (torch.Tensor) : target flow
        inpt   (torch.Tensor) : predicted flow
        
"""


def epe_loss(input, target, reduction=None):

    # u, v components
    u_i, v_i = input[:, 0], input[:, 1]
    u_t, v_t = target[:, 0], target[:, 1]

    # Euclidean distance
    d = torch.sqrt((u_i - u_t)**2 + (v_i - v_t)**2)

    # return mean distance
    if reduction == "sum":
        d = torch.sum(d)
    elif reduction == "element_wise_mean":
        d = d.mean()

    return d


"""
Cosine Similarity Loss for flow vectors

    Args:
        target (torch.Tensor) : target flow
        inpt   (torch.Tensor) : predicted flow
"""


def cosine_loss(input, target, reduction=None):

    # cosine distance between each vector
    d = 1.0 - f.cosine_similarity(input, target, dim=1)

    if reduction == "sum":
        d = torch.sum(d)
    elif reduction == "element_wise_mean":
        d = torch.mean(d)

    return d
