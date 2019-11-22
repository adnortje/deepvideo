# imports
import torch.nn as nn
import modules.functional as f

"""
Pixel Shuffle 2D

    increases video frame resolution (depth-to-space unit)

        Args:
            upscale_factor (int) : factor to increase spatial resolution and decrease channel depth by

        Ref:
            https://arxiv.org/abs/1609.05158
"""


class PixelShuffle2D(nn.Module):

    def __init__(self, upscale_factor):
        super(PixelShuffle2D, self).__init__()

        self.upscale_factor = upscale_factor

    def forward(self, x):
        return f.pixel_shuffle_2d(x, self.upscale_factor)

    def extra_repr(self):
        return 'upscale_factor={}'.format(self.upscale_factor)

