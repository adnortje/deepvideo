# imports
import torch
import torch.nn as nn
from modules import ImportanceMap
from .video_encoder import VideoEncoder
from .video_decoder import VideoDecoder

"""
Video Autoencoder Network

    B-frame Video Autoencoder that encodes & binarizes / decodes a set of B-video frames

"""


class VideoAuto(nn.Module):

    def __init__(self, bnd, stateful=False):
        super(VideoAuto, self).__init__()

        # model name
        self.name = "VideoAuto"
        self.display_name = "Video Autoencoder"

        # bottleneck-depth
        self.bnd = bnd

        # encoder network
        self.encoder = VideoEncoder(
            bnd=self.bnd
        )

        # decoder network
        self.decoder = VideoDecoder(
            bnd=self.bnd,
            stateful=stateful
        )

    def encode(self, x):
        vid_feat_map, b = self.encoder(x)

        if hasattr(self, "imp_map"):
            b, p = self.imp_map(b, feat_map=vid_feat_map)
        else:
            p = None

        return b, p

    def decode(self, x, dec_states):
        x = self.decoder(x, dec_states)
        return x

    def forward(self, x, dec_state=None):
        # encode & decode x
        vid_feat_map, b = self.encode(x)
        x = self.decode(b, dec_state)
        return x

    def encode_decode(self, x, dec_state=None):
        # decode
        b, _ = self.encode(x)
        x = self.decode(b, dec_state)
        return x, b

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
