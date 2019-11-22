# import
import torch
import torch.nn as nn
from .if_feat_ext import IFrameFeatExtractor


"""
I-Frame Feature Extractor

"""


class TwinIFrameFeatExtractor(nn.Module):

    def __init__(self):
        super(TwinIFrameFeatExtractor, self).__init__()

        # I-Frame Feature Extractors
        self.iframe_feat_0 = IFrameFeatExtractor()
        self.iframe_feat_t = IFrameFeatExtractor()

        # 1x1 Convolutions
        self.conv_512 = nn.Sequential(
            nn.Conv2d(
                in_channels=1024,
                out_channels=512,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                bias=True
            ),
            nn.ReLU()
        )

        self.conv_256 = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=256,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                bias=True
            ),
            nn.ReLU()
        )

        self.conv_128 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=128,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                bias=True
            ),
            nn.ReLU()
        )

        self.conv_64 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=64,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                bias=True
            ),
            nn.ReLU()
        )

    def forward(self, x):
        # get I-Frames
        iframe_0 = x[:, :, 0, :, :].unsqueeze(2)
        iframe_t = x[:, :, -1, :, :].unsqueeze(2)

        # extract I-Frame features
        i_feat_0 = self.iframe_feat_0(iframe_0)
        i_feat_t = self.iframe_feat_t(iframe_t)

        # combined feature dictionary
        i_feat = {

            "512": self.conv_512(
                torch.cat(
                    (i_feat_0["512"], i_feat_t["512"]),
                    dim=1
                )
            ).unsqueeze(2),

            "256": self.conv_256(
                torch.cat(
                    (i_feat_0["256"], i_feat_t["256"]),
                    dim=1
                )
            ).unsqueeze(2),

            "128": self.conv_128(
                torch.cat(
                    (i_feat_0["128"], i_feat_t["128"]),
                    dim=1
                )
            ).unsqueeze(2),

            "64": self.conv_64(
                torch.cat(
                    (i_feat_0["64"], i_feat_t["64"]),
                    dim=1
                )
            ).unsqueeze(2)
        }

        return i_feat
