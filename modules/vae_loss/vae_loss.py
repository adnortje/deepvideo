# imports
import torch.nn as nn
import modules.functional as f

"""
VAE loss

    combines a reconstruction and KLD loss term.

    Args:
        r_loss (string) : criteria to calculate reconstruction loss [MSE or BCE]
        beta (int)      : factor to scale KLD loss by. Drive up for disentangled encodings

    Ref:
        https://openreview.net/forum?id=Sy2fzU9gl Beta-VAE Appendix Beta-norm
"""


class VAELoss(nn.Module):

    def __init__(self, r_loss, beta=1.0):
        super(VAELoss, self).__init__()

        # Reconstruction Loss Criteria

        if r_loss == 'BCE':
            # Bernoulli decoder
            self.r_loss = nn.BCELoss(
                size_average=False
            )

        elif r_loss == 'MSE':
            # Gaussian decoder
            self.r_loss = nn.MSELoss(
                size_average=False
            )

        else:
            raise ValueError('Reconstruction Loss not in [MSE, BCE]')

        # Beta term
        self.beta = beta

    def forward(self, inpt, target, mu, logvar):

        # Summed Reconstruction Loss divided by batch size
        r_loss = self.r_loss(inpt, target=target) / inpt.size(0)

        # averaged KLD loss
        kld = f.kld_loss_gaussian(mu, logvar, reduction="element_wise_mean")

        return r_loss + self.beta * kld
