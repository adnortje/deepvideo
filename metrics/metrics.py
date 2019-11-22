# import
import re
import torch
import numpy as np
from math import exp
import torch.nn as nn
import tempfile as tmp
import skvideo.io as sk
import subprocess as sb
from modules.liteflownet import EvalFlow, DenseFlow

"""
Check that reference & compressed image tenor dimensions are equal

"""


def _assert_compatible(r_img, c_img):
    if r_img.size() != c_img.size():
        # check images have equal dimensions
        raise ValueError('Reference & compressed image dimension are not equal!')

    return


"""
Check images adhere to specified range

"""


def _assert_valid_data_range(r_img, c_img, data_range):

    # actual min and max values
    c_range = torch.max(c_img) - torch.min(c_img)
    r_range = torch.max(r_img) - torch.min(r_img)

    if r_range > data_range:
        raise ValueError("Reference image has intensity values outside the range specified by data_range.")

    if c_range > data_range:
        raise ValueError("Compressed image has intensity values outside the range specified by data_range.")

    return


"""
Structural SIMilarity index(SSIM)

    Function used to compute the SSIM between two Tensor images

        Args:
            r_img (torch.Tensor)    : reference or original video (T, C, H, W)
            c_img (torch.Tensor)    : compressed video (T, C, H, W)
            data_range (int)        : dynamic allowed range between r_img & c_img
            multichannel (bool)     : average SSIM over each image channel
            gaussian_weights (bool) : use gaussian weights
            

        Returns:
            ssim (float): comparative SSIM score between two videos [-1, 1]

        Ref:
            http://www.cns.nyu.edu/pub/lcv/wang03-preprint.pdf
            https://github.com/scikit-image/scikit-image/blob/v0.14.1/skimage/measure/_structural_similarity.py#L13
            https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py
"""


class SSIM:

    def __init__(
            self,
            data_range,
            win_size=None, gradient=False,
            multichannel=False, gaussian_weights=False,
            full=False, **kwargs):

        self.full = full
        self.gradient = gradient
        self.data_range = data_range
        self.multichannel = multichannel
        self.gaussian_weights = gaussian_weights

        if win_size is None:

            if gaussian_weights:
                self.win_size = 11
            else:
                self.win_size = 7

        else:
            self.win_size = win_size

        if self.win_size % 2 != 1:
            raise ValueError('win_size must be odd')

        # filter parameters

        self.k1 = kwargs.pop('K1', 0.01)
        self.k2 = kwargs.pop('K2', 0.03)

        if self.k1 < 0:
            raise ValueError("K1 must be positive")
        if self.k2 < 0:
            raise ValueError("K2 must be positive")

        self.sigma = kwargs.pop('sigma', 1.5)

        if self.sigma < 0:
            raise ValueError("sigma must be positive")

        self.c1 = (self.k1 * data_range) ** 2
        self.c2 = (self.k2 * data_range) ** 2

        self.use_sample_covariance = kwargs.pop('use_sample_covariance', True)

    def calc_video(self, r_vid, c_vid):

        if isinstance(r_vid, np.ndarray):
            # convert numpy inputs to torch tensors
            r_vid = torch.from_numpy(r_vid).permute(0, 3, 1, 2)
            c_vid = torch.from_numpy(c_vid).permute(0, 3, 1, 2)

            # cast to type torch.FloatTensor
            r_vid = r_vid.type(torch.FloatTensor)
            c_vid = c_vid.type(torch.FloatTensor)

        # check dimensions
        _assert_compatible(r_vid, c_vid)

        # check data range
        _assert_valid_data_range(r_vid, c_vid, self.data_range)

        # calculate ssim
        ssim = self._calc_ssim(r_vid, c_vid)

        return ssim

    def _calc_ssim(self, x, y):

        if self.multichannel:
            # calc ssim for all channels
            n_ch = x.size(1)
        else:
            # use only luma channel
            x = x[:, 0, :, :].unsqueeze(1)
            y = y[:, 0, :, :].unsqueeze(1)
            n_ch = x.size(1)

        if x.size(2) < self.win_size or x.size(3) < self.win_size:
            raise ValueError('win_size greater than image extent')

        if self.gaussian_weights:
            # gaussian filter
            filter_func = self._gaussian_filter(n_ch)

        else:
            # uniform filter
            filter_func = self._uniform_filter(n_ch)

        # filter on same device as input
        filter_func = filter_func.to(x.device)

        # window size x channels
        np = self.win_size ** len(x[0].size())

        if self.use_sample_covariance:
            cov_norm = np / (np - 1)
        else:
            cov_norm = 1.0

        # compute means
        ux = filter_func(x)
        uy = filter_func(y)

        # compute variances and covariances
        uxx = filter_func(x * x)
        uyy = filter_func(y * y)
        uxy = filter_func(x * y)
        vx = cov_norm * (uxx - ux * ux)
        vy = cov_norm * (uyy - uy * uy)
        vxy = cov_norm * (uxy - ux * uy)

        a1, a2, b1, b2 = (
            2 * ux * uy + self.c1,
            2 * vxy + self.c2,
            ux ** 2 + uy ** 2 + self.c1,
            vx + vy + self.c2
        )

        d = b1 * b2
        ssim_map = (a1 * a2) / d

        # ssim_map cropped to avoid padding effect
        i = self.win_size // 2
        ssim_map_crop = ssim_map[:, :, i:ssim_map.size(2)-i, i:ssim_map.size(3)-i]

        # mean ssim
        mssim = ssim_map_crop.mean()

        if self.gradient:
            grad = filter_func(a1 / d) * x
            grad += filter_func(-ssim_map / b2) * y
            grad += filter_func((ux * (a2 - a1) - uy * (b2 - b1) * ssim_map) / d)
            grad *= (2 / x[0].numel())

            if self.full:
                return mssim, ssim_map, grad
            else:
                return mssim, grad

        if self.full:
            return mssim, ssim_map

        return mssim

    def _gaussian_filter(self, channels):
        # create gaussian filter

        # gaussian weights
        gaussian_kernel = torch.Tensor([
            exp(
                -(i - self.win_size//2)**2/float(2*self.sigma**2)
            ) for i in range(self.win_size)
        ])

        # normalize weights
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

        # 1D kernel
        gaussian_kernel = gaussian_kernel.unsqueeze(1)

        # 2D kernel
        gaussian_kernel = gaussian_kernel.mm(
            gaussian_kernel.t()
        ).unsqueeze(0).unsqueeze(0)

        # reshape weights
        gaussian_kernel = gaussian_kernel.view(
            1,
            1,
            self.win_size,
            self.win_size
        ).expand(
            channels, 1, self.win_size, self.win_size
        ).contiguous()

        # filter
        gaussian_filter = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=self.win_size,
            groups=channels,
            padding=self.win_size // 2,
            bias=False
        )

        # set conv gaussian kernel weights
        gaussian_filter.weight.data = gaussian_kernel

        # don't require grad
        gaussian_filter.weight.requires_grad = False

        return gaussian_filter

    def _uniform_filter(self, channels):
        # create uniform filter

        # uniform filter kernel
        uniform_kernel = torch.full(
            size=(channels, 1, self.win_size, self.win_size),
            fill_value=1/self.win_size**2
        )

        # filter
        uniform_filter = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=self.win_size,
            groups=channels,
            padding=self.win_size // 2,
            bias=False
        )

        # set conv gaussian kernel weights
        uniform_filter.weight.data = uniform_kernel

        # don't require grad
        uniform_filter.weight.requires_grad = False

        return uniform_filter

    @staticmethod
    def check_threshold(thresh):
        if thresh < -1.0 or thresh > 1.0:
            raise ValueError("SSIM Threshold out of bounds!")
        return


"""
Peak Signal to Noise Ratio (PSNR)

    Function used to compute the Peak-Signal-to-Noise-Ratio (PSNR) between two Tensor videos

        Args:
            r_vid (torch.Tensor) : reference video (T, C, H, W)
            c_vid (torch.Tensor) : compressed video (T, C, H, W)
            data_range (int)     : dynamic allowed range between r_img & c_img

        Returns:
            psnr (float): comparative PSNR between the two images in dB

        Ref:
            https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
"""


class PSNR:

    def __init__(self, data_range):
        self.name = 'PSNR'
        self.data_range = data_range

    def calc_video(self, r_vid, c_vid):

        if isinstance(r_vid, np.ndarray):
            # convert numpy inputs to torch tensors
            r_vid = torch.from_numpy(r_vid).permute(0, 3, 1, 2)
            c_vid = torch.from_numpy(c_vid).permute(0, 3, 1, 2)

            # cast to type torch.FloatTensor
            r_vid = r_vid.type(torch.FloatTensor)
            c_vid = c_vid.type(torch.FloatTensor)

        # check images compatible
        _assert_compatible(r_vid, c_vid)

        # check data_range is valid
        _assert_valid_data_range(r_vid, c_vid, self.data_range)

        # calculate psnr
        psnr = self._calc_psnr(r_vid, c_vid)

        return psnr

    def _calc_psnr(self, x, y):

        psnr = []

        for t in range(x.size(0)):
            # mean-square-error (MSE)
            mse = ((x[t] - y[t])**2).mean()

            # peak-signal-to-noise-ratio (PSNR)
            psnr.append(10*torch.log10((self.data_range**2)/mse))

        psnr = (sum(psnr) / len(psnr)).item()

        return psnr


"""
EPE 

    Function used to compute the End-to-end Point Error (EPE) between video optical flows

        Args:
            r_flow (torch.Tensor) : reference video optical flow (T, C, H, W)
            c_flow (torch.Tensor) : compressed video optical flow (T, C, H, W)

        Returns:
            epe (float): comparative EPE between the two videos

        Ref:
            https://stackoverflow.com/questions/49699739/what-is-endpoint-error-between-optical-flows
"""


class EPE:

    def __init__(self, standard=False):

        self.name = "EPE"
        self.standard = standard

        if standard:
            self.flow_net = DenseFlow()
        else:
            self.flow_net = EvalFlow()

    def calc_video(self, r_vid, c_vid):

        if self.standard:
            r_flow = self.standard_flow(r_vid)
            c_flow = self.standard_flow(c_vid)

        else:
            r_flow = self.deep_flow(r_vid)
            c_flow = self.deep_flow(c_vid)

        # calculate epe
        epe = self._calc_epe(r_flow, c_flow)

        return epe

    def deep_flow(self, vid):
        # Optical Flow using LiteFlowNet

        if isinstance(vid, np.ndarray):
            # numpy -> torch.Tensor
            vid = torch.from_numpy(vid).permute(3, 0, 1, 2)
            vid = vid.type(torch.FloatTensor)
        else:
            # (C, T, H, W) -> (T, C, H, W)
            vid = vid.permute(1, 0, 2, 3)

        flow = self.flow_net(vid.unsqueeze(0))
        flow = flow[0].permute(1, 0, 2, 3)

        return flow

    def standard_flow(self, vid):
        # Optical Flow using Farneback Polynomial Expansion

        if isinstance(vid, torch.Tensor):
            # torch.Tensor -> Numpy
            vid = vid.numpy().transpose(0, 2, 3, 1)

        vid = vid.astype(np.float32)

        flow = self.flow_net(vid)

        return flow

    @staticmethod
    def _calc_epe(x, y):

        epe = 0.0

        for t in range(x.size(0)):
            # u & v vectors
            ux, vx = x[t, 0], x[t, 1]
            uy, vy = y[t, 0], y[t, 1]
            # calc epe at each time-step
            d = torch.sqrt((ux - uy) ** 2 + (vx - vy) ** 2).mean()
            epe += d

        # avg across frames
        epe /= x.size(0)
        return epe

    @staticmethod
    def _normalize_flow(flow, init_h, init_w):
        flow[:, 0] /= init_w
        flow[:, 1] /= init_h
        return flow


"""
Video Multi-Method Assessment Fusion (VMAF)

    Function used to compute the VMAF score between two Tensor videos

        Args:
            r_vid (torch.Tensor) : (0, 1) normalized reference video (T, C, H, W)
            c_vid (torch.Tensor) : (0, 1) compressed video (T, C, H, W)

        Returns:
            vmaf (float): VMAF score between the two videos [0-100]

        Ref:
            https://github.com/Netflix/vmaf
        
        Note: 
            assumes ffmpeg installation with VMAF enabled
"""


class VMAF:

    def __init__(self):
        self.name = "VMAF"

    def calc_video(self, r_vid, c_vid):

        if isinstance(r_vid, np.ndarray):
            # convert numpy inputs to torch tensors; pre-empt future full torch implementation
            r_vid = torch.from_numpy(r_vid).permute(0, 3, 1, 2)
            c_vid = torch.from_numpy(c_vid).permute(0, 3, 1, 2)

        # check videos are compatible
        _assert_compatible(r_vid, c_vid)

        # calculate VMAF score
        vmaf = self._calc_vmaf(r_vid, c_vid)

        return vmaf

    def _calc_vmaf(self, r_vid, c_vid):

        # create temp video files
        r_tmp = tmp.NamedTemporaryFile(suffix='.mp4')
        c_tmp = tmp.NamedTemporaryFile(suffix='.mp4')

        # write to temp files
        self._write_vf(r_vid, r_tmp.name)
        self._write_vf(c_vid, c_tmp.name)

        # ffmpeg cmd
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-i",
            r_tmp.name,
            "-i",
            c_tmp.name,
            "-lavfi",
            "libvmaf",
            "-f",
            "null",
            "-",
        ]

        buff = sb.Popen(cmd, stdout=sb.PIPE)

        # find VMAF score
        vmaf = float(
            re.search("VMAF score = (\d+\.\d+)", str(buff.stdout.read())).group(1)
        )

        # close & destroy temp files
        r_tmp.close()
        c_tmp.close()

        return vmaf

    @staticmethod
    def _write_vf(tensor_vid, fp):
        # GPU -> CPU
        tensor_vid = tensor_vid.cpu()
        # torch.Tensor -> Numpy
        np_vid = tensor_vid.numpy() * 255.0
        np_vid = np_vid.astype(np.uint8)
        # (T, C, H, W) -> (T, H, W, C)
        np_vid = np_vid.transpose(0, 2, 3, 1)
        # write video file
        sk.vwrite(fp, np_vid)
        return
