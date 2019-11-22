# imports
import re
import os
import tempfile
import subprocess as sb
import skvideo.io as sk
from metrics import PSNR, SSIM

"""
Class FFmpeg 
    
    class uses ffmpeg to compress a video file using specified codec
    
    Args:
        codec (string) : video codec to compress video file with
        n_gop (int)    : GOP length
        
    Note:
        f_s (tuple) : frame size is width then height
"""


class FFmpeg:
    
    def __init__(self, codec, n_gop):
        # def codec & gop size
        self.codec = codec
        self.n_gop = n_gop
        
    def encode(self, video, q=0, ret_vid=True, save_loc="/tmp"):
        # encode and save file
        save_loc = os.path.expanduser(save_loc)

        if not os.path.isdir(save_loc):
            raise NotADirectoryError("Save location d.n.e!")
        
        # create file (.mp4 container)
        ffmpeg_cmd = None

        if self.codec in  ["libvpx-vp9"]:
            vid_ext = ".webm"
        else:
            vid_ext = ".mp4"

        fn = self.codec + str(q) + vid_ext
        fl = open(fn, "w")
        
        if self.n_gop % 2 == 0:
            # use bounding I-frames
            g = self.n_gop - 1
        else:
            # I-Frame at start only
            g = self.n_gop

        if self.codec in ["libx264"]:
            # encode video
            ffmpeg_cmd = [
                "ffmpeg",
                "-loglevel", "quiet",
                "-y",
                "-i", video,
                "-codec:v", self.codec,
                "-g", str(g),
                "-keyint_min", str(g),
                "-crf", str(q),
                "-an",
                "-f", "mp4",
                fl.name
            ]

        elif self.codec in ["libvpx-vp9"]:
            # encode video
            ffmpeg_cmd = [
                "ffmpeg",
                "-loglevel", "quiet",
                "-y",
                "-i", video,
                "-codec:v", self.codec,
                "-g", str(g),
                "-keyint_min", str(g),
                "-minrate", str(q),
                "-maxrate", str(q),
                "-b:v", str(q),
                "-an",
                "-f", "webm",
                fl.name
            ]

        elif self.codec in ["libx265"]:
            # encode video
            ffmpeg_cmd = [
                "ffmpeg",
                "-loglevel", "quiet",
                "-y",
                "-i", video,
                "-codec:v", self.codec,
                "-x265-params", "'keyint="+str(g)+":min-keyint="+str(g)+"'",
                "-crf", str(q),
                "-frames:v", str(self.n_gop),
                "-an",
                "-f", "mp4",
                fl.name
            ]

        # ffmpeg subprocess call
        sb.call(ffmpeg_cmd)

        if ret_vid:
            vid = sk.vread(fl.name)
            return vid

        return

    def encode_img(self, img, q=0):
        # encode and save file

        tmp = tempfile.NamedTemporaryFile(suffix='.png')

        # encode video
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-loglevel", "quiet",
            "-i", img,
            "-codec:v", self.codec,
            "-crf", str(q),
            "-frames:v", str(self.n_gop),
            tmp.name
        ]

        # ffmpeg subprocess call
        sb.call(ffmpeg_cmd)

        return tmp
        
    def _encode(self, video, q=0, vf=None):
        # encode & return handle to tmp file
        ffmpeg_cmd = None
        tmp = vf

        if self.n_gop % 2 == 0:
            # use bounding I-frames
            g = self.n_gop - 1
        else:
            # I-Frame at start only
            g = self.n_gop

        if self.codec in ["libx264"]:
            #tmp = tempfile.NamedTemporaryFile(suffix='.mp4')
            # encode video
            ffmpeg_cmd = [
                "ffmpeg",
                "-loglevel", "quiet",
                "-y",
                "-i", video,
                "-codec:v", self.codec,
                "-g", str(g),
                "-keyint_min", str(g),
                "-crf", str(q),
                "-frames:v", str(self.n_gop),
                "-an",
                "-f", "mp4",
                tmp
            ]

        elif self.codec in ["libvpx-vp9"]:
            #tmp = tempfile.NamedTemporaryFile(suffix='.webm')
            # encode video
            ffmpeg_cmd = [
                "ffmpeg",
                "-loglevel", "quiet",
                "-y",
                "-i", video,
                "-codec:v", self.codec,
                "-g", str(g),
                "-keyint_min", str(g),
                "-crf", str(q),
                "-b:v", "500",
                "-frames:v", str(self.n_gop),
                "-an",
                "-f", "webm",
                tmp
            ]

        elif self.codec in ["libx265"]:
            #tmp = tempfile.NamedTemporaryFile(suffix='.mp4')
            # encode video
            ffmpeg_cmd = [
                "ffmpeg",
                "-loglevel", "quiet",
                "-y",
                "-i", video,
                "-codec:v", self.codec,
                "-x265-params", "'keyint="+str(g)+":min-keyint="+str(g)+"'",
                "-crf", str(q),
                "-frames:v", str(self.n_gop),
                "-an",
                "-f", "mp4",
                tmp
            ]

        # ffmpeg subprocess call
        sb.call(ffmpeg_cmd)
        
        return tmp

    def _calc_bpp(self, video):
        # calculate video bits-per-pixel
        meta = sk.ffprobe(video)['video']

        # read video metadata
        w = float(meta['@width'])
        h = float(meta['@height'])

        # file_size in bits
        bits = os.path.getsize(video)*8

        # calc bpp
        bpp = bits / (self.n_gop * w * h)
        
        return bpp

    @staticmethod
    def _calc_ssim_ffmpeg(r_vid, c_vid):
        # calculate SSIM
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-i", c_vid,
            "-i", r_vid,
            "-lavfi",
            "[0:v][1:v]ssim",
            "-f", "null",
            "-"
        ]
        buff = sb.Popen(cmd, stderr=sb.PIPE)
        
        # find ssim
        ssim = float(
            re.search(
                'Y:(\d+\.\d+)',
                str(buff.stderr.read()),
                re.IGNORECASE
            ).group(1)
        )

        return ssim

    @staticmethod
    def _calc_ssim(r_vid, c_vid):
        # calculate SSIM Guassian

        np_r_vid = sk.vread(r_vid)
        np_c_vid = sk.vread(c_vid)

        ssim = SSIM(
            data_range=255,
            multichannel=True,
            gaussian_weights=True
        ).calc_video(np_r_vid, np_c_vid)

        return ssim

    @staticmethod
    def _calc_psnr(r_vid, c_vid):
        # calculate PSNR

        np_r_vid = sk.vread(r_vid)
        np_c_vid = sk.vread(c_vid)

        psnr = PSNR(
            data_range=255
        ).calc_video(np_r_vid, np_c_vid)

        return psnr

    @staticmethod
    def _calc_psnr_ffmpeg(r_vid, c_vid):
        # calc PSNR
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "quiet",
            "-i", c_vid,
            "-i", r_vid,
            "-lavfi", "[0:v][1:v]psnr",
            "-f", "null",
            "-"
        ]
        buff = sb.Popen(cmd, stderr=sb.PIPE)
        # find psnr
        psnr = float(
            re.search(
                'average:(\d+\.\d+)',
                str(buff.stderr.read()),
                re.IGNORECASE
            ).group(1)
        )

        return psnr

    @staticmethod
    def _calc_vmaf(r_vid, c_vid):
        # calc VMAF
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "quiet",
            "-i", r_vid,
            "-i", c_vid,
            "-lavfi", "libvmaf",
            "-f",
            "null",
            "-",
        ]
        buff = sb.Popen(cmd, stdout=sb.PIPE)

        # find VMAF score
        vmaf = float(
            re.search(
                "VMAF score = (\d+\.\d+)",
                str(buff.stdout.read())
            ).group(1)
        )

        return vmaf

    def calc_ssim_bpp(self, video, q=0, ssim_ffmpeg=False):
        # calc SSIM at quality factor
        tmp = self._encode(video, q)
        
        # calc ssim & bpp 
        bpp = self._calc_bpp(tmp.name)

        if ssim_ffmpeg:
            ssim = self._calc_ssim_ffmpeg(video, tmp.name)
        else:
            ssim = self._calc_ssim(video, tmp.name)
        
        # close tmp file
        tmp.close()
        
        return ssim, bpp
             
    def calc_psnr_bpp(self, video, q=0, psnr_ffmpeg=False):
        # calc PSNR at quality factor
        tmp = self._encode(video, q)
        
        # calc psnr & bpp 
        bpp = self._calc_bpp(tmp.name)

        if psnr_ffmpeg:
            psnr = self._calc_psnr_ffmpeg(video, tmp.name)
        else:
            psnr = self._calc_psnr(video, tmp.name)
        
        # close vid file
        tmp.close()

        return psnr, bpp

    def calc_vmaf_bpp(self, video, q=0):
        # calc VMAF at quality factor
        tmp = self._encode(video, q)
        # calc vmaf & bpp
        bpp = self._calc_bpp(tmp.name)
        vmaf = self._calc_vmaf(video, tmp.name)

        # close vid file
        tmp.close()

        return vmaf, bpp

    def resize_video(self, video, f_s):
        # encode & return handle to tmp file
        tmp = tempfile.NamedTemporaryFile(suffix='.mp4')

        # encode video
        ffmpeg_cmd = [
            "ffmpeg",
            "-loglevel", "quiet",
            "-y",
            "-i", video,
            "-vf", "scale =" + str(f_s[0])+":"+str(f_s[1]),
            "-frames:v", str(self.n_gop),
            tmp.name
        ]

        # ffmpeg subprocess call
        sb.call(ffmpeg_cmd)
        return tmp
