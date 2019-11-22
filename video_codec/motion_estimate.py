# imports
import numpy as np
from utils import img_t, vid_t
import skvideo.motion as sk_m
from metrics import SSIM, PSNR, VMAF, EPE
import torchvision.transforms as tf
from timeit import default_timer as timer
from process_data import NpFrame2PIL, VideoDataset, CropVideoSequence
from skimage import measure
"""
Motion Estimation & Compensation Evaluation Class

"""


class EvalMVC(object):

    def __init__(
            self,
            video_dir,
            vid_ext="mp4",
            frame_size=(224, 320), num_frames=24,
            method="DS", mb_size=16, search_dist=7):

        # per frame transform
        frame_transform = tf.Compose([
            NpFrame2PIL("RGB"),
            tf.Resize(frame_size)
        ])

        # composed transform
        video_transform = tf.Compose([
            CropVideoSequence(
                num_frames=num_frames
            ),
            tf.Lambda(
                lambda frames: np.stack(
                    [frame_transform(frame) for frame in frames]
                )
            )
        ])

        # check video directory
        self.video_dataset = VideoDataset(
            root_dir=video_dir,
            vid_ext=vid_ext,
            transform=video_transform
        )

        self.num_videos = len(self.video_dataset)

        # motion parameters
        self.frame_size = frame_size
        self.num_frames = num_frames
        self.method = method
        self.mb_size = mb_size
        self.search_dist = search_dist

        # evaluation metrics

        # SSIM
        self.ssim = SSIM(
            data_range=255,
            multichannel=True,
            gaussian_weights=True,
        )

        # EPE using LiteFLowNet
        self.epe = EPE(standard=False)

        # PSNR
        self.psnr = PSNR(
            data_range=255
        )

        # VMAF
        self.vmaf = VMAF()

    def avg_time(self):
        # average vector estimation and compensation time (sec)

        total_time = 0.0

        for r_vid in self.video_dataset:
            # sum time
            start_time = timer()
            self.ipp_bmc(r_vid, self.ipp_bme(r_vid))[1:]
            end_time = timer()
            total_time += end_time - start_time

        avg_time = total_time / self.num_videos

        return avg_time

    def avg_vmaf(self):
        # average VMAF

        total_vmaf = 0.0

        for r_vid in self.video_dataset:
            # sum vmaf values
            total_vmaf += self.calc_vmaf(
                r_vid[1:] / 255,
                self.ipp_bmc(r_vid, self.ipp_bme(r_vid))[1:] / 255
            )

        avg_vmaf = total_vmaf / self.num_videos

        return avg_vmaf

    def avg_ssim(self):
        # average SSIM
        total_ssim = 0.0

        for r_vid in self.video_dataset:

            # sum ssim values
            total_ssim += self.calc_ssim(
                r_vid[1:],
                self.ipp_bmc(r_vid, self.ipp_bme(r_vid))[1:]
            )

        avg_ssim = total_ssim / self.num_videos

        return avg_ssim

    def avg_psnr(self):
        # average PSNR

        total_psnr = 0.0

        for r_vid in self.video_dataset:
            # sum psnr values
            total_psnr += self.calc_psnr(
                r_vid[1:],
                self.ipp_bmc(r_vid, self.ipp_bme(r_vid))[1:]
            )

        avg_psnr = total_psnr / self.num_videos

        return avg_psnr

    def avg_epe(self):
        # average EPE

        total_epe = 0.0

        for r_vid in self.video_dataset:
            # sum epe values
            total_epe += self.calc_epe(
                r_vid / 255,
                self.ipp_bmc(r_vid, self.ipp_bme(r_vid)) / 255
            )

        avg_epe = total_epe / self.num_videos

        return avg_epe

    def avg_bpp(self):
        # average bits-per-pixel

        total_bpp = 0.0

        for r_vid in self.video_dataset:
            # sum bpp values
            total_bpp += self.calc_bpp(
                self.ipp_bme(r_vid)
            )

        avg_bpp = total_bpp / self.num_videos

        return avg_bpp

    def ipp_bme(self, videodata):
        # I, P, P, P Block Motion Estimation
        motion = sk_m.blockMotion(
            videodata,
            method=self.method,
            mbSize=self.mb_size,
            p=self.search_dist
        )
        # motion (numFrames - 1, height / mbSize, width / mbSize, 2)
        return motion

    def ipp_bmc(self, videodata, motion):
        # I, P, P, P Block Motion Compensation
        bmc = sk_m.blockComp(
            videodata,
            motionVect=motion,
            mbSize=self.mb_size
        )

        return bmc

    def display_avg_stats(self):
        # print averaged scores
        print("Bpp  : {}".format(self.avg_bpp()))
        print("PSNR : {}".format(self.avg_psnr()))
        print("SSIM : {}".format(self.avg_ssim()))
        print("VMAF : {}".format(self.avg_vmaf()))
        print("EPE  : {}".format(self.avg_epe()))
        print("Time (sec) : {}".format(self.avg_time()))
        return

    def display_bmc_video(self, index=0):
        # display Block Motion Compensated Video
        r_vid = self.video_dataset[index]
        c_vid = self.ipp_bmc(r_vid, self.ipp_bme(r_vid))

        # print evaluation metrics
        bpp_str = "bpp : {}".format(
            round(self.calc_bpp(self.ipp_bme(r_vid)), 4)
        )
        psnr_str = "PSNR : {}".format(
            round(self.calc_psnr(r_vid[1:], c_vid[1:]), 2)
        )
        ssim_str = "SSIM : {}".format(
            round(self.calc_ssim(r_vid[1:], c_vid[1:]), 2)
        )
        vmaf_str = "VMAF : {}".format(
            round(self.calc_ssim(r_vid[1:] / 255, c_vid[1:] / 255), 2)
        )

        # set-up plot
        x_label = "".join([psnr_str, ssim_str, vmaf_str])
        img_t.setup_plot("", y_label=bpp_str, x_label=x_label)

        # display compensated sequence
        vid_t.display_frames(c_vid / 255)

        return

    def calc_vmaf(self, r_vid, c_vid):
        # calculate VMAF
        return self.vmaf.calc_video(r_vid, c_vid)

    def calc_ssim(self, r_vid, c_vid):
        # calculate SSIM
        return self.ssim.calc_video(r_vid, c_vid)

    def calc_psnr(self, r_vid, c_vid):
        # calculate PSNR
        return self.psnr.calc_video(r_vid, c_vid)

    def calc_epe(self, r_vid, c_vid):
        # calculate EPE
        return self.epe.calc_video(r_vid, c_vid)

    def calc_bpp(self, motion_vectors):
        # calculate bpp for Motion Vectors
        # Note: this is direct binarisation without overhead for retaining shape
        # i.e. how many bpp do we need to convey motion

        total_bits = 0.0

        t, h, w, _ = motion_vectors.shape

        for f in range(t):
            for y in range(h):
                for x in range(w):
                    dx, dy = motion_vectors[f, y, x]

                    if dy != 0.0 or dx != 0.0:
                        total_bits += self.bit_count(dy) + self.bit_count(dx) + self.bit_count(x) + self.bit_count(y)
        # bits per pixel
        f_h, f_w = self.frame_size
        bpp = total_bits / (f_h * f_w * t)

        return bpp

    def calc_cc(self, metric, save_dir="./"):
        # calculate compression curve

        if metric not in ["PSNR", "SSIM", "VMAF", "EPE"]:
            raise KeyError("Specified metric : {}, is not currently supported!".format(metric))

        # calculate metric values
        met, bpp = self._prog_eval(metric)

        # compression curve dictionary
        curve = {"bpp": bpp, "metric": met}

        # create file name
        file_name = "".join([save_dir, "/", self.method, "_", metric, '.npy'])

        # save curve as numpy file
        np.save(file_name, curve)

    def _prog_eval(self, metric):

        # metric & bpp lists
        m = []
        b = []

        # macro-block sizes
        og_mb_size = self.mb_size
        mb_sizes = [4, 8, 16]

        for mb_size in mb_sizes:

            self.mb_size = mb_size

            if metric == "PSNR":
                m_val = self.avg_psnr()

            elif metric == "SSIM":
                m_val = self.avg_ssim()

            elif metric == "VMAF":
                m_val = self.avg_vmaf()

            elif metric == "EPE":
                m_val = self.avg_epe()
            else:
                m_val = None

            b_val = self.avg_bpp()

            # append values
            m.append(m_val)
            b.append(b_val)

        # reset macro-block size to original
        self.mb_size = og_mb_size

        return m, b

    @staticmethod
    def bit_count(val):
        # return number of bits needed to represent val
        return len(np.binary_repr(int(val)))

