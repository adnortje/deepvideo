# imports
import math
import torch
import torch.nn as nn
import torch.nn.functional as f
from .feature import Features
from .matching import Matching
from .subpixel import Subpixel
from .regularization import Regularization


class LiteFlowNet(nn.Module):

    def __init__(self):
        super(LiteFlowNet, self).__init__()

        self.moduleFeatures = Features()

        self.moduleMatching = nn.ModuleList([
            Matching(intLevel) for intLevel in [2, 3, 4, 5, 6]
        ])

        self.moduleSubpixel = nn.ModuleList([
            Subpixel(intLevel) for intLevel in [2, 3, 4, 5, 6]
        ])

        self.moduleRegularization = nn.ModuleList([
            Regularization(intLevel) for intLevel in [2, 3, 4, 5, 6]
        ])

    def forward(self, tensorFirst, tensorSecond):

        # Pre-Process
        tensorFirst, init_h, init_w = self.prep(tensorFirst)
        tensorSecond, _, _ = self.prep(tensorSecond)

        # Normalization
        tensorFirst[:, 0, :, :] = tensorFirst[:, 0, :, :] - 0.411618
        tensorFirst[:, 1, :, :] = tensorFirst[:, 1, :, :] - 0.434631
        tensorFirst[:, 2, :, :] = tensorFirst[:, 2, :, :] - 0.454253

        tensorSecond[:, 0, :, :] = tensorSecond[:, 0, :, :] - 0.410782
        tensorSecond[:, 1, :, :] = tensorSecond[:, 1, :, :] - 0.433645
        tensorSecond[:, 2, :, :] = tensorSecond[:, 2, :, :] - 0.452793

        tensorFeaturesFirst = self.moduleFeatures(tensorFirst)
        tensorFeaturesSecond = self.moduleFeatures(tensorSecond)


        tensorFirst = [tensorFirst]
        tensorSecond = [tensorSecond]

        for intLevel in [1, 2, 3, 4, 5]:
            tensorFirst.append(
                f.interpolate(
                    input=tensorFirst[-1],
                    size=(tensorFeaturesFirst[intLevel].size(2), tensorFeaturesFirst[intLevel].size(3)),
                    mode='bilinear',
                    align_corners=False
                )
            )

            tensorSecond.append(
                f.interpolate(
                    input=tensorSecond[-1],
                    size=(tensorFeaturesSecond[intLevel].size(2), tensorFeaturesSecond[intLevel].size(3)),
                    mode='bilinear',
                    align_corners=False
                )
            )

        tensorFlow = None

        for intLevel in [-1, -2, -3, -4, -5]:
            tensorFlow = self.moduleMatching[intLevel](
                tensorFirst[intLevel],
                tensorSecond[intLevel],
                tensorFeaturesFirst[intLevel],
                tensorFeaturesSecond[intLevel],
                tensorFlow
            )

            tensorFlow = self.moduleSubpixel[intLevel](
                tensorFirst[intLevel],
                tensorSecond[intLevel],
                tensorFeaturesFirst[intLevel],
                tensorFeaturesSecond[intLevel],
                tensorFlow
            )

            tensorFlow = self.moduleRegularization[intLevel](
                tensorFirst[intLevel],
                tensorSecond[intLevel],
                tensorFeaturesFirst[intLevel],
                tensorFeaturesSecond[intLevel],
                tensorFlow
            )

        # Post Processing
        tensorFlow = self.post(tensorFlow * 20, init_h, init_w)

        return tensorFlow

    @staticmethod
    def post(p, init_h, init_w):

        # prep height and width
        prep_h = int(math.floor(math.ceil(init_h / 32.0) * 32.0))
        prep_w = int(math.floor(math.ceil(init_w / 32.0) * 32.0))

        p = f.interpolate(
            input=p,
            size=(init_h, init_w),
            mode='bilinear',
            align_corners=False
        )

        # clone to fix backward error
        p[:, 0] = p[:, 0].clone() * float(init_w / prep_w)
        p[:, 1] = p[:, 1].clone() * float(init_h / prep_h)

        return p

    @staticmethod
    def prep(p):

        init_h, init_w = p.size()[2:4]

        prep_h = int(math.floor(math.ceil(init_h / 32.0) * 32.0))
        prep_w = int(math.floor(math.ceil(init_w / 32.0) * 32.0))

        # [-1, 1] -> [0, 1]
        p = (p * 0.5) + 0.5

        p = f.interpolate(
                input=p,
                size=(prep_h, prep_w),
                mode='bilinear',
                align_corners=False
        )

        return p, init_h, init_w

    def load_model(self, save_loc):
        # load model weights
        self.load_state_dict(torch.load(save_loc))
        return
