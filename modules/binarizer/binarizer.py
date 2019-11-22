# imports
import torch
import torch.nn as nn
from functions import Binarize

"""
Binarizer Module
    
"""


class Binarizer(nn.Module):

    def __init__(self):
        super(Binarizer, self).__init__()

    def forward(self, x):
        # [-1, 1] -> {-1, 1}
        x = Binarize.apply(x, self.training)
        return x

