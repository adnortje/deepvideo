# imports
import torch
import torch.nn.functional as f
from .var import Backward_tensorGrid

"""
Backward Function

"""


def Backward(tensorInput, tensorFlow):

    if str(tensorFlow.size()) not in Backward_tensorGrid:
        tensorHorizontal = torch.linspace(
            -1.0, 1.0, tensorFlow.size(3)
        ).view(
            1, 1, 1, tensorFlow.size(3)
        ).expand(
            tensorFlow.size(0), -1, tensorFlow.size(2), -1
        )

        tensorVertical = torch.linspace(
            -1.0, 1.0, tensorFlow.size(2)
        ).view(
            1, 1, tensorFlow.size(2), 1
        ).expand(
            tensorFlow.size(0), -1, -1, tensorFlow.size(3)
        )

        Backward_tensorGrid[str(tensorFlow.size())] = torch.cat((tensorHorizontal, tensorVertical), 1).cuda()

    tensorFlow = torch.cat(
        (
            tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0),
            tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0)
        ), 1
    )

    ret = f.grid_sample(
        input=tensorInput,
        grid=(Backward_tensorGrid[str(tensorFlow.size())] + tensorFlow).permute(0, 2, 3, 1),
        mode='bilinear',
        padding_mode='zeros'
    )

    return ret
