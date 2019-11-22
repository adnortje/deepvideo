# imports
import torch
import torch.nn as nn
from .network import LiteFlowNet

# Pre-trained LiteFlowNet Models
SIN = "./modules/liteflownet/saved_models/network-sintel.pytorch"
KIT = "./modules/liteflownet/saved_models/network-kitti.pytorch"
DEF = "./modules/liteflownet/saved_models/network-default.pytorch"

"""
FlowNet

    Calculate MSE between estimated optical flow of hallucinated and input video frames

    Args:
        flow_model (string) : path to saved LiteFlowNet model weights

"""


class EvalFlow(nn.Module):

    def __init__(self, flow_model=SIN):
        super(EvalFlow, self).__init__()

        if not torch.cuda.is_available():
            raise NotImplementedError("LiteFlowNet only implemented for CUDA")

        # LiteFlowNet
        self.flow_net = LiteFlowNet()
        # load model weights
        self.flow_net.load_model(save_loc=flow_model)
        # Eval mode & GPU
        self.eval().cuda()

        # Fix pre-trained weights
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, vid):

        # [0, 1] -> [-1, 1]
        vid = (vid - 0.5) / 0.5

        # only implemented for CUDA
        vid = vid.cuda()
        _, _, _, h, w = vid.size()

        flow_list = []

        for t in range(vid.size(2) - 1):
            # input & target flow
            flow = self.flow_net(vid[:, :, t], vid[:, :, t + 1])
            flow_list.append(flow)

        flow = torch.stack(flow_list, dim=2)

        return flow
