# imports
import torch.nn as nn
from modules.vae_loss import VAELoss


"""
IterativeLoss

    Penalize system for reconstruction loss at each model iteration
    Note: assumes first iteration is a predictive VAE network

    Args:
        r_loss (string) : criteria to calculate reconstruction loss [MSE, BCE]

"""


class IterativeLoss(nn.Module):

    def __init__(self, r_loss, vae_beta=1.0):

        super(IterativeLoss, self).__init__()

        if r_loss == "BCE":
            # VAE Loss
            self.vae_loss = VAELoss(
                r_loss="BCE",
                beta=vae_beta
            )
            # Reconstruction Loss
            self.r_loss = nn.BCE()

        elif r_loss == "MSE":
            # VAE Loss
            self.vae_loss = VAELoss(
                r_loss="MSE",
                beta=vae_beta
            )
            # Reconstruction Loss
            self.r_loss = nn.MSELoss()

    def forward(self, *args, target, mu, logvar):

        losses = []
        args = list(args)

        # VAE Loss
        pred = args.pop(0)
        loss = self.vae_loss(pred, target=target, mu=mu, logvar=logvar)
        losses.append(loss)

        for arg in args:
            # append iterative losses
            loss = self.r_loss(arg, target=target)
            losses.append(loss)

        # sum & normalize iterative losses
        loss = sum(losses) / len(losses)

        return loss
