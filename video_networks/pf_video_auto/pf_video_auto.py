# imports
import torch
import torch.nn as nn
from modules import ImportanceMap
from .pf_decoder import PFrameDecoder
from .video_encoder import VideoEncoder
from .if_feat_ext import IFrameFeatExtractor
from .ms_video_encoder import MultiScaleVideoEncoder

"""
P-Frame Video Autoencoder Network

"""


class PFrameVideoAuto(nn.Module):

    def __init__(self, bnd, multiscale=False):
        super(PFrameVideoAuto, self).__init__()

        # model name
        self.name = "PFrameVideoAuto"
        self.display_name = "P-Frame Video Autoencoder"

        # bottleneck-depth
        self.bnd = bnd

        # I-Frame Feature Extractor
        self.iframe_feat_0 = IFrameFeatExtractor()

        # Video Encoder Network
        if multiscale:
            self.video_encoder = MultiScaleVideoEncoder(
                bnd=bnd
            )
        else:
            self.video_encoder = VideoEncoder(
                bnd=bnd
            )

        # B-Frame Decoder Network
        self.bframe_decoder = PFrameDecoder(
            bnd=bnd
        )

    def encode(self, x):
        vid_feat_map, b = self.video_encoder(x)

        if hasattr(self, "imp_map"):
            b, p = self.imp_map(b, feat_map=vid_feat_map)
        else:
            p = None

        return b, p

    def decode(self, x, i_frame):
        x = self.bframe_decoder(x, i_frame)
        return x

    def encode_decode(self, x):
        # I-Frame features
        iframe_0 = x[:, :, 0, :, :].unsqueeze(2)
        i_feat_0 = self.iframe_feat_0(iframe_0)

        # encode movement
        b, _ = self.encode(x)

        # decode using I-Frame
        x = self.decode(b, i_feat_0)

        return x, b

    def forward(self, x):

        # I-Frame features
        iframe_0 = x[:, :, 0, :, :].unsqueeze(2)
        i_feat_0 = self.iframe_feat_0(iframe_0)

        # encode movement
        b, p = self.encode(x)

        # decode using I-Frame
        x = self.decode(b, i_feat_0)

        if hasattr(self, "imp_map"):
            return x, p

        return x

    def load_model(self, save_loc):
        # load model weights
        self.load_state_dict(torch.load(save_loc))
        return

    def fine_tune_bitrate(self, L):
        # add importance map module
        self.L=L
        self.imp_map = ImportanceMap(
            input_dim=512,
            n=self.bnd,
            L=L
        )
        return
