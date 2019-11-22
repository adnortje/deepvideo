# imports
import torch.nn as nn
from modules.vae_loss import VAELoss

"""
VAE Interpolation Loss

    combines VAE  loss and interpolation loss

    Args:
        r_loss (string) : criteria to calculate reconstruction loss [MSE or BCE]
        beta (int)      : factor to scale KLD loss by. Drive up for disentangled encodings

    Ref:
        https://openreview.net/forum?id=Sy2fzU9gl Beta-VAE Appendix Beta-norm
"""


class VAEIntpLoss(nn.Module):

    def __init__(self, r_loss, beta=1.0):
        super(VAEIntpLoss, self).__init__()

        # VAE Loss
        self.vae_loss = VAELoss(r_loss, beta)

        # Interpolation Reconstruction Loss
        if r_loss == 'BCE':
            # Bernoulli decoder
            self.intp_loss = nn.BCELoss()

        elif r_loss == 'MSE':
            # Gaussian decoder
            self.intp_loss = nn.MSELoss()

        else:
            raise ValueError('Reconstruction Loss not in [MSE, BCE]')

    def forward(self, inpt, target, mu, logvar, intp, target_intp):

        # VAE LOSS
        vae_loss = self.vae_loss(
            inpt,
            target=target,
            mu=mu,
            logvar=logvar
        )

        # Interpolation Loss
        intp_loss = self.intp_loss(intp, target=target_intp)

        return vae_loss + intp_loss
