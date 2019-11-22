# imports
import os
import numpy as np
from .ffmpeg import FFmpeg
from utils import img_t, vid_t
from process_data.video_codec_dataset import VideoCodecDataset

"""
Video Codec
    
    evaluate compression performed by standard video codecs
    
    Args:
        codec    (string) : name of lossy video codec
        root_dir (string) : video directory 
"""


class VideoCodec:
    
    def __init__(self, codec, n_gop, f_s, root_dir, vid_ext='mp4'):

        root_dir = os.path.expanduser(root_dir)

        if not os.path.isdir(root_dir):
            raise NotADirectoryError("Specified directory d.n.e!")

        # codec
        self.codec = codec
        self.video_coder = FFmpeg(
            codec=codec,
            n_gop=n_gop
        )

        # video frame size
        self.f_s = f_s

        # def video dataset
        self.vid_set = VideoCodecDataset(
            root_dir=root_dir,
            vid_ext=vid_ext
        )

    def disp_comp_frames(self, q=0, fr_start=0, fr_end=-1):
        # display compressed frames
        v = self.vid_set.__getitem__(0)
        v_tmp = self.video_coder.resize_video(v, self.f_s)

        # scores & bpp
        y_lb = self._get_bpp_str(v_tmp, q)
        x_lb = self._get_score_str(v_tmp, q)

        # encoded video
        cv = self.video_coder.encode(v_tmp.name, q, ret_vid=True)

        fig = img_t.setup_plot("", y_label=y_lb, x_label=x_lb)

        # display compressed frames
        vid_t.display_frames(cv[fr_start:fr_end])
        fig.savefig("vid.pdf")

        return

    def _get_score_str(self, r_vid, q):
        # returns a string of metric scores
        psnr, _ = self.video_coder.calc_psnr_bpp(r_vid.name, q)
        ssim, _ = self.video_coder.calc_ssim_bpp(r_vid.name, q)
        vmaf, _ = self.video_coder.calc_vmaf_bpp(r_vid.name, q)

        psnr_str = "PSNR : {}".format(round(psnr, 2))
        ssim_str = "SSIM : {}".format(round(ssim.item(), 3))
        vmaf_str = "VMAF : {}".format(round(vmaf, 2))

        stats_str = "\n".join([psnr_str, ssim_str, vmaf_str])
        return stats_str

    def _get_bpp_str(self, r_vid, q):
        # return bpp string
        _, bpp = self.video_coder.calc_psnr_bpp(r_vid.name, q)
        bpp_str = "bpp : {}".format(round(bpp, 4))
        return bpp_str
                  
    def create_cc(self, metric, save_loc="./"):
        # create and save compression curve
        cc = {'met': [], 'bpp': []}

        if self.video_coder.codec == "libvpx-vp9":
            MAX_CRF = 0
            Q_STEP = -3
            Q_START = 63
        else:
            MAX_CRF = 0
            Q_STEP = 3
            Q_START = 52

        for q in range(Q_START, MAX_CRF, Q_STEP):
            
            t_met = 0.0
            t_bpp = 0.0

            for v in self.vid_set:

                v_tmp = self.video_coder.resize_video(v, self.f_s)

                if metric == "PSNR":
                    met, bpp = self.video_coder.calc_psnr_bpp(v_tmp.name, q)
                elif metric == "SSIM":
                    met, bpp = self.video_coder.calc_ssim_bpp(v_tmp.name, q)
                elif metric == "VMAF":
                    met, bpp = self.video_coder.calc_vmaf_bpp(v_tmp.name, q)

                # add metric & bitrate
                t_met += met
                t_bpp += bpp

                # close and delete temp file
                v_tmp.close()

            # append averaged metric & bpp
            cc['met'].append(t_met / len(self.vid_set))
            cc['bpp'].append(t_bpp / len(self.vid_set))
        
        # save curve as numpy dict
        fn = save_loc + "/" + self.codec + "_" + metric + ".npy"
        np.save(fn, cc)

        return
