# imports
import torch
import torch.nn as nn
from .network import LiteFlowNet
from modules.functional import epe_loss, cosine_loss

# Pre-trained LiteFlowNet Models
SIN = "./modules/liteflownet/saved_models/network-sintel.pytorch"
KIT = "./modules/liteflownet/saved_models/network-kitti.pytorch"
DEF = "./modules/liteflownet/saved_models/network-default.pytorch"

"""
LiteFlowNetLoss

    Calculate MSE between estimated optical flow of hallucinated and input video frames

    Args:
    
    Refs:
        https://arxiv.org/pdf/1805.07036.pdf
        https://github.com/sniklaus/pytorch-liteflownet.git
        https://github.com/twhui/LiteFlowNet

"""


class LiteFlowNetLoss(nn.Module):

    def __init__(self, flow_model=SIN, flow_loss="EPE"):
        super(LiteFlowNetLoss, self).__init__()

        if not torch.cuda.is_available():
            raise NotImplementedError("LiteFlowNet only implemented for CUDA")

        # LiteFlowNet
        self.flow_net = LiteFlowNet()
        self.flow_net.load_model(save_loc=flow_model)
        self.eval().cuda()

        # Fix pre-trained weights
        for param in self.parameters():
            param.requires_grad = False

        if flow_loss not in ["EPE", "COSINE"]:
            raise KeyError("Specified flow loss; {}, is not currently supported!".format(flow_loss))
        else:
            self.flow_loss = flow_loss

    def forward(self, inpt, target):

        losses = []

        if inpt.size() != target.size():
            raise ValueError("Input and target sizes are not equal!")

        # extract width and height
        _, _, _, h, w = inpt.size()

        # only implemented for CUDA
        inpt = inpt.cuda()
        targ = target.cuda()

        inpt_flows = []
        targ_flows = []

        for i in range(inpt.size(2)-1):

            # input & target flow
            inpt_flow = self.flow_net(inpt[:, :, i], inpt[:, :, i+1])
            targ_flow = self.flow_net(targ[:, :, i], targ[:, :, i+1])

            inpt_flows.append(inpt_flow)
            targ_flows.append(targ_flow)

        inpt_flow = torch.stack(inpt_flows, dim=2)
        targ_flow = torch.stack(targ_flows, dim=2)

        # normalize flow in x and y directions
        inpt_flow[:, 0] = inpt_flow[:, 0].clone() / h
        inpt_flow[:, 1] = inpt_flow[:, 1].clone() / w
        targ_flow[:, 0] = targ_flow[:, 0].clone() / h
        targ_flow[:, 1] = targ_flow[:, 1].clone() / w

        # calculate loss
        if self.flow_loss == "EPE":
            loss = epe_loss(inpt_flow, target=targ_flow, reduction="element_wise_mean")
        elif self.flow_loss == "COSINE":
            loss = cosine_loss(inpt_flow, target=targ_flow, reduction="element_wise_mean")

        return loss
