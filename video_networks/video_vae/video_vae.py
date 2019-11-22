# imports
import torch
import torch.nn as nn
from .vae_dec import VideoVAEDecoder
from .vae_enc import VideoVAEEncoder

"""
Video VAE Network with Encoder Motion Conditioning

"""


class VideoVAE(nn.Module):

    def __init__(self, bnd, vae_bnd=128):
        super(VideoVAE, self).__init__()

        # model name
        self.name = "VideoVAE"
        self.display_name = "VideoVAE"

        # bottleneck-depth
        self.bnd = bnd

        # Video VAE Encoder Network
        self.encoder = VideoVAEEncoder(
            bnd=self.bnd,
            vae_bnd=vae_bnd
        )

        # Video VAE Decoder Network
        self.decoder = VideoVAEDecoder(
            bnd=vae_bnd
        )

    def encode(self, x, i_f):
        # encode mean & variance
        mu, logvar = self.encoder(x, iframe=i_f)

        # draw z samples
        z = self._reparameterize(mu, logvar)

        return z, mu, logvar

    def decode(self, z):
        # decode latent space
        x = self.decoder(z)
        return x

    def forward(self, x):

        # GOP
        n_gop = x.size(2)

        # I-Frame at t0
        i_fr_0 = x[:, :, 0, :, :]

        # conditioning frames
        cond_fr = x[:, :, range(2, n_gop, n_gop//8)]

        # encode
        z, mu, logvar = self.encode(cond_fr, i_fr_0)

        # decode
        fr_t = self.decode(z)

        return fr_t, mu, logvar

    def interpolate(self, x):

        # GOP
        b, c, n_gop, h, w = x.size()

        # I-Frame at t0
        i_fr_0 = x[:, :, 0, :, :]

        # conditioning frames
        cond_fr_0 = i_fr_0.unsqueeze(dim=2).expand(b, c, 8, h, w)
        cond_fr_t = x[:, :, range(2, n_gop, n_gop//8)]

        # encode
        z_0, mu_0, logvar_0 = self.encode(cond_fr_0, i_fr_0)
        z_t, mu_t, logvar_t = self.encode(cond_fr_t, i_fr_0)

        # difference vector
        diff_z = z_t - z_0

        # interpolated frames
        intp_fr = []

        for n in range(1, n_gop-1):
            # decode latent vectors
            intp_fr.append(self.decode(z_0 + n * diff_z / (n_gop - 1)))

        intp_fr = torch.stack(intp_fr, dim=2)

        return intp_fr

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
