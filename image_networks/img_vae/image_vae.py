# imports
import torch
import torch.nn as nn
from .vae_enc import VAEEncoder
from .vae_dec import VAEDecoder

"""
Image VAE Network

"""


class ImageVAE(nn.Module):

    def __init__(self, bnd):
        super(ImageVAE, self).__init__()

        self.name = "ImageVAE"
        self.display_name = "ImageVAE"

        # bottleneck-depth
        self.bnd = bnd

        # encoder & decoder networks
        self.encoder = VAEEncoder(
            bnd=bnd
        )

        # decoder network
        self.decoder = VAEDecoder(
            bnd=bnd
        )

    def encode(self, x):
        # encode mean & variance
        mu, logvar = self.encoder(x)
        # draw z samples
        z = self._reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, x):
        # decode using I-Frame Features
        x = self.decoder(x)
        return x

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        x = self.decode(z)
        return x, mu, logvar

    @staticmethod
    def _reparameterize(mu, logvar):
        # reparameterization trick
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size(), device=std.device)
        z = mu + std * esp
        return z

    def load_model(self, save_loc):
        # load model weights
        self.load_state_dict(torch.load(save_loc))
        return
