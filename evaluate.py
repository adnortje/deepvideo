# imports
import copy
import torch
import numpy as np
from utils import img_t, vid_t
from timeit import default_timer as timer
from video_codec import ImageCodec
from metrics import SSIM, PSNR, EPE, VMAF
from modules.liteflownet import EvalFlow, DenseFlow

"""
Class EvalVideoModel
 
    used to evaluate performance of a Video Compression Model
  
        Args:
            model        (nn.Module)  : trained video compression model 
            dataloaders  (DataLoader) : video dataLoader dictionary
            standard_epe (boolean)    : LiteFLowNet : False or FarnebackFlow : True
    
    Note: Eval currently only supports non-NVVL dataLoaders 
        
"""


class EvalVideoModel:
    
    def __init__(self, model, dataloaders, inc_overhead=False, if_codec=None, standard_epe=False):

        # use GPU if available
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        # model to device & inference mode
        self.model = model.to(self.device)
        self.model.train(False)

        # video dataloaders
        vid_dls = dataloaders
        self.f_s = vid_dls.f_s

        self.n_gop = vid_dls.n_gop
        if "PFrame" in self.model.name:
            # remove reference frame
            self.n_gop = self.n_gop -1
        elif "BFrame" in self.model.name:
            # remove reference frames
            self.n_gop = self.n_gop - 2

        self.vid_dls = vid_dls.get_data_loaders()

        # I-Frame image codec
        self.if_codec = if_codec
        if if_codec is not None:
            self.img_codec = ImageCodec(
                codec=if_codec
            )

        # include overhead bits
        self.inc_overhead = inc_overhead

        # evaluation metrics

        # SSIM
        self.ssim = SSIM(
            data_range=1,
            multichannel=True,
            gaussian_weights=True,
        )

        # PSNR
        self.psnr = PSNR(
            data_range=1
        )

        # EPE using Farneback or LiteFlowNet
        self.epe = EPE(
            standard=standard_epe
        )

        self.standard_epe = standard_epe

        # VMAF
        self.vmaf = VMAF()

    def compare_frames(self, dataset="valid", widget=False):

        # load nxt GOP
        gop = iter(self.vid_dls[dataset]).next()

        # compress GOP
        c_gop, r_gop = self._predict_frames(gop)

        # display frames
        if widget:
            vid_t.vs_display_frames_widget(r_gop[0], c_gop[0])
        else:
            vid_t.vs_display_frames(r_gop[0], c_gop[0])

        # display evaluation metric scores
        self.disp_stats(r_gop[0], c_gop[0])

        return

    def disp_pred_frames(self, dataset="valid", widget=False):
        # plots predicted frames with quality scores
        gop = iter(self.vid_dls[dataset]).next()

        # predict frames
        c_gop, r_gop = self._predict_frames(gop)

        # set-up plot
        y_lb = self._get_bpp_str(r_gop)
        x_lb = self._get_score_str(r_gop[0], c_gop[0])
        img_t.setup_plot("", y_label=y_lb, x_label=x_lb)

        # display predicted frames
        vid_t.display_frames(c_gop[0])

        return

    def disp_comp_frames(self, dataset="valid", q=0, frames_start=0, frame_end=-1):
        # plots compressed frames
        gop = iter(self.vid_dls[dataset]).next()

        # predict frames
        c_gop = self._compress_frames(gop, q)

        # setup plot
        y_lb = self._get_bpp_str(gop, q)
        x_lb = self._get_score_str(gop[0], c_gop[0])
        fig = img_t.setup_plot("", y_label=y_lb, x_label=x_lb)

        # display compressed frames
        vid_t.display_frames(c_gop[0, frames_start:frame_end])
        fig.savefig('vid.pdf')

        return

    def _get_score_str(self, r_vid, c_vid):
        # returns a string of metric scores
        psnr_str = "PSNR : {}".format(round(self.calc_psnr(r_vid, c_vid), 2))
        ssim_str = "SSIM : {}".format(round(self.calc_ssim(r_vid, c_vid).item(), 3))
        vmaf_str = "VMAF : {}".format(round(self.calc_vmaf(r_vid, c_vid), 2))
        stats_str = "\n".join([psnr_str, ssim_str, vmaf_str])
        return stats_str

    def _get_bpp_str(self, r_vid, q=None):
        # return bpp string
        bpp_str = "bpp : {}".format(round(self.calc_bpp(r_vid, q), 4))
        return bpp_str

    def disp_stats(self, r_vid, c_vid):
        # display evaluation metric scores
        print("Bpp  : {}".format(self.calc_bpp(r_vid, c_vid)))
        print("PSNR : {}".format(self.calc_psnr(r_vid, c_vid)))
        print("SSIM : {}".format(self.calc_ssim(r_vid, c_vid)))
        print("VMAF : {}".format(self.calc_vmaf(r_vid, c_vid)))

        if self.model.name != "ImageVAE":
            # Flow only works for video networks
            print("EPE  : {}".format(self.calc_epe(r_vid, c_vid)))

        return

    def disp_avg_stats(self, dataset):
        # display average compression statistics
        print("PSNR : {}".format(self.avg_psnr(dataset)))
        print("SSIM : {}".format(self.avg_ssim(dataset)))
        print("VMAF : {}".format(self.avg_vmaf(dataset)))
        print("Time : {}".format(self.avg_time(dataset)))
        print("Bpp  : {}".format(self.avg_bpp(dataset)))

        if self.model.name != "ImageVAE":
            # Flow only works for video networks
            print("EPE  : {}".format(self.avg_epe(dataset)))

        return

    def calc_cc(self, metric, dataset="valid", save_loc="./"):
        # calculate compression curve

        met = []
        bpp = []

        for q in range(1, 52, 3):
            # vary I-Frame quantisation
            m = self.avg_met(metric, dataset, q)
            b = self.avg_bpp(dataset, q)
            met.append(m)
            bpp.append(b)

        cc = {"met": met, "bpp": bpp}

        np.save(save_loc+self.model.name + " ("+ self.if_codec+")_" + metric+".npy", cc)

        return cc

    def avg_met(self, metric, dataset, q=None):
        # calculate average score for given metric

        met = None

        if metric not in ["PSNR", "SSIM", "VMAF", "EPE"]:
            raise ValueError("Specified metric: {}, is not currently supported!".format(metric))

        # calculate chosen metric
        if metric == "PSNR":
            met = self.avg_psnr(dataset, q)
        elif metric == "SSIM":
            met = self.avg_ssim(dataset, q)
        elif metric == "VMAF":
            met = self.avg_vmaf(dataset, q)
        elif met == "EPE":
            met = self.avg_epe(dataset, q)

        return met

    def avg_ssim(self, dataset="valid", q=None):
        # average SSIM score for dataset
        total_ssim = 0.0

        for r_vid in self.vid_dls[dataset]:
            # sum SSIM

            if q is None:
                c_vid, r_vid = self._predict_frames(r_vid)
            else:
                c_vid = self._compress_frames(r_vid, q)

            total_ssim += self.calc_ssim(r_vid[0], c_vid[0])

        avg_ssim = total_ssim / len(self.vid_dls[dataset])

        return avg_ssim

    def avg_psnr(self, dataset="valid", q=None):
        # average PSNR score for dataset

        total_psnr = 0.0

        for r_vid in self.vid_dls[dataset]:
            # sum PSNR

            if q is None:
                c_vid, r_vid = self._predict_frames(r_vid)
            else:
                c_vid = self._compress_frames(r_vid, q)

            total_psnr += self.calc_psnr(r_vid[0], c_vid[0])

        avg_psnr = total_psnr / len(self.vid_dls[dataset])

        return avg_psnr

    def avg_epe(self, dataset="valid"):
        # average EPE score for dataset
        total_epe = 0.0

        for r_vid in self.vid_dls[dataset]:
            # sum EPE
            c_vid, _ = self._predict_frames(r_vid)

            # include first frame motion
            if "PFrame" in self.model.name:
                c_vid = torch.cat(
                    (r_vid[:, 0].unsqueeze(1), c_vid),
                    dim=1
                )
            elif "BFrame" or "MotionCond" in self.model.name:
                c_vid = torch.cat(
                    (r_vid[:, 0].unsqueeze(1), c_vid, r_vid[:, -1].unsqueeze(1)),
                    dim=1
                )

            total_epe += self.calc_epe(r_vid[0], c_vid[0])

        avg_epe = total_epe / len(self.vid_dls[dataset])

        return avg_epe

    def avg_vmaf(self, dataset="valid", q=None):
        # average VMAF score for dataset
        total_vmaf = 0.0

        for r_vid in self.vid_dls[dataset]:
            # sum VMAF
            if q is None:
                c_vid, r_vid = self._predict_frames(r_vid)
            else:
                c_vid = self._compress_frames(r_vid, q)

            total_vmaf += self.calc_vmaf(r_vid[0], c_vid[0])

        avg_vmaf = total_vmaf / len(self.vid_dls[dataset])

        return avg_vmaf

    def avg_bpp(self, dataset="valid", q=None):
        # average Bpp for dataset
        total_bpp = 0.0

        for r_vid in self.vid_dls[dataset]:
            # motion bits
            total_bpp += self.calc_bpp(r_vid, q)

        avg_bpp = total_bpp / len(self.vid_dls[dataset])

        return avg_bpp

    def avg_time(self, dataset=None):
        # average encoding & decoding time
        total_time = 0.0

        for r_vid in self.vid_dls[dataset]:
            # sum time
            start_time = timer()
            self._predict_frames(r_vid)
            end_time = timer()
            total_time += end_time - start_time

        avg_time = total_time / len(self.vid_dls[dataset])

        return avg_time

    def calc_ssim(self, r_vid, c_vid):
        # calculate SSIM
        return self.ssim.calc_video(r_vid, c_vid)

    def calc_psnr(self, r_vid, c_vid):
        # calculate PSNR
        return self.psnr.calc_video(r_vid, c_vid)

    def calc_vmaf(self, r_vid, c_vid):
        # calculate VMAF
        return self.vmaf.calc_video(r_vid, c_vid)

    def calc_epe(self, r_vid, c_vid):
        # calculate EPE
        return self.epe.calc_video(r_vid, c_vid)

    def calc_bpp(self, r_vid, q=None):
        bpp = None
        # motion bits
        b, p = self._encode_frames(r_vid)
        pred_bits = b.view(-1).size(0)

        if p is not None and self.inc_overhead:
            # add overhead bits
            pred_bits += p.view(-1).size(0)

        if q is not None:
            # add I-Frame bits
            _, i_bits = self._encode_i_frame(r_vid, q)
            pred_bits += i_bits

        if self.model.name == "PFrameVideoAuto":
            bpp = pred_bits / ((self.n_gop+1) * self.f_s[0] * self.f_s[1])

        elif self.model.name == "BFrameVideoAuto":
            bpp = pred_bits / ((self.n_gop+2) * self.f_s[0] * self.f_s[1])

        return bpp

    def calc_bits(self, r_vid, q=None, both=False):

        # motion bits
        b, p = self._encode_frames(r_vid)
        pred_bits = b.view(-1).size(0)

        if p is not None and self.inc_overhead:
            # add overhead bits
            pred_bits += p.view(-1).size(0)

        if q is not None:
            # add I-Frame bits
            i_bits = self._i_frame_bits(r_vid, q, both)
            pred_bits += i_bits

        return pred_bits

    def _i_frame_bits(self, r_gop, q, both=False):
        # encode I-Frames using image codec

        r_gop = copy.deepcopy(r_gop)

        r_gop[0, 0], i_bits = self.img_codec.encode_decode(r_gop[0, 0], q)

        if both:
            r_gop[0, -1], i2_bits = self.img_codec.encode_decode(r_gop[0, -1], q)
            i_bits += i2_bits

        return i_bits

    def disp_bit_heatmaps(self, dataset="valid", widget=False):
        # display bitrate heat maps
        gop = iter(self.vid_dls[dataset]).next()

        # get heat-map
        h_map = self._get_heatmap(gop)

        vid_t.display_heatmap(h_map[0, 0])
        return

    def disp_flow(self, dataset="valid", widget=False):
        # display input vs output optical flow

        # LiteFlowNet
        if self.standard_epe:
            flow_net = DenseFlow()
        else:
            flow_net = EvalFlow()

        # load next GOP
        gop = iter(self.vid_dls[dataset]).next()

        # compress GOP
        c_gop, r_gop = self._comp_frames(gop)

        if self.standard_epe:
            r_gop = (r_gop[0]).numpy()
            c_gop = (c_gop[0]).cpu().numpy()
            r_gop = r_gop.transpose(0, 2, 3, 1)
            c_gop = c_gop.transpose(0, 2, 3, 1)
        else:
            # (B, D, C, H, W) -> (B, C, D, H, W)
            r_gop = (r_gop - 0.5) / 0.5
            c_gop = (c_gop - 0.5) / 0.5
            r_gop = r_gop.permute(0, 2, 1, 3, 4)
            c_gop = c_gop.permute(0, 2, 1, 3, 4)

        # input and output optical flow
        inp_flow = flow_net(r_gop)
        out_flow = flow_net(c_gop)

        if not self.standard_epe:
            # (B, C, D, H, W) -> (B, D, C, H, W)
            inp_flow = inp_flow.permute(0, 2, 1, 3, 4).cpu()[0]
            out_flow = out_flow.permute(0, 2, 1, 3, 4).cpu()[0]

        if widget:
            vid_t.vs_display_flow_widget(inp_flow, out_flow)
        else:
            vid_t.vs_display_flow(inp_flow, out_flow)

        return

    def _get_heatmap(self, r_gop):
        # encode video frames and return heat-maps

        with torch.no_grad():

            # place on GPU
            r_gop = r_gop.to(self.device)

            # (B, D, C, H, W) -> (B, C, D, H, W)
            r_gop = r_gop.permute(0, 2, 1, 3, 4)

            # normalise
            norm_gop = (r_gop - 0.5) / 0.5

            _, p = self.model.encode(norm_gop)

        return p

    def _encode_i_frame(self, r_gop, q):
        # encode I-Frames using image codec

        r_gop = copy.deepcopy(r_gop)

        r_gop[0, 0], i_bits = self.img_codec.encode_decode(r_gop[0, 0], q)

        if "BFrame" in self.model.name:
            r_gop[0, -1], i2_bits = self.img_codec.encode_decode(r_gop[0, -1], q)
            i_bits += i2_bits

        return r_gop, i_bits

    def _encode_frames(self, r_gop):
        # encode video frames to bits

        with torch.no_grad():

            # place on GPU
            r_gop = r_gop.to(self.device)

            # (B, D, C, H, W) -> (B, C, D, H, W)
            r_gop = r_gop.permute(0, 2, 1, 3, 4)

            # normalise
            norm_gop = (r_gop - 0.5) / 0.5

            b, p = self.model.encode(norm_gop)

            # get rid of masked bits
            b = b[b != 0]

            if p is not None:
                # binarize importance map
                p = self._binarize_imp_map(p)

        return b, p

    def _decode_frames(self, b, gop):
        # predict frames from bits
        i_feat = self.iframe_feat(gop)
        dec = self.model.decode(b, i_feat)
        return dec

    def _binarize_imp_map(self, p):
        # quantise importance map
        pq = torch.floor(self.model.L * p)

        pq = np.unpackbits(
            pq.cpu().numpy().astype(np.uint8)
        )

        # remove unnecessary bits
        pq = pq.reshape(-1, 8)
        if self.model.L > 2:
            pq = pq[:, self._bit_count(self.model.L):]
        pq = pq.reshape(-1)

        return torch.tensor(pq)

    @staticmethod
    def _bit_count(val):
        # return number of bits needed to represent val
        return len(np.binary_repr(int(val)))

    def _compress_frames(self, r_gop, q):

        with torch.no_grad():

            # place on GPU
            r_gop = r_gop.to(self.device)

            # normalise
            norm_gop = (r_gop - 0.5) / 0.5

            # (B, D, C, H, W) -> (B, C, D, H, W)
            norm_gop = norm_gop.permute(0, 2, 1, 3, 4)

            # encode
            b, _ = self.model.encode(norm_gop)

            # encode & decode I-Frames
            c_gop, _ = self._encode_i_frame(r_gop.cpu(), q)

            if self.model.name in ["PFrameVideoAuto"]:
                i_feat = self.model.iframe_feat_0(
                    (c_gop.permute(0, 2, 1, 3, 4)[:, :, 0, :, :].unsqueeze(2).to(self.device) - 0.5) / 0.5
                )
            elif self.model.name in ["BFrameVideoAuto"]:
                i_feat = self.model.iframe_feat(
                    (c_gop.permute(0, 2, 1, 3, 4).to(self.device) - 0.5) / 0.5
                )

            # decode predicted frames
            dec = self.model.decode(b, i_feat)

            # (B, C, D, H, W) -> (B, D, C, H, W)
            dec = dec.permute(0, 2, 1, 3, 4)

            # inverse normalization
            dec = (dec * 0.5) + 0.5

            if self.model.name in ["PFrameVideoAuto"]:
                c_gop[:, 1:] = dec
            elif self.model.name in ["BFrameVideoAuto"]:
                c_gop[:, 1:-1] = dec

            # back to CPU
            c_gop = c_gop.cpu()

            return c_gop

    def _predict_frames(self, r_gop):
        # compress video frames

        with torch.no_grad():

            # place on GPU
            r_gop = r_gop.to(self.device)

            # (B, D, C, H, W) -> (B, C, D, H, W)
            r_gop = r_gop.permute(0, 2, 1, 3, 4)

            # normalise
            norm_gop = (r_gop - 0.5) / 0.5

            # compress GOP
            if self.model.name == "ImageVAE":
                # only single image compression
                c_gop, _, _ = self.model(norm_gop[:, :, 0])
                c_gop = c_gop.unsqueeze(2)

            else:
                c_gop, _ = self.model.encode_decode(norm_gop)

            # inverse normalization
            c_gop = (c_gop * 0.5) + 0.5

            # (B, C, D, H, W) -> (B, D, C, H, W)
            r_gop = r_gop.permute(0, 2, 1, 3, 4)
            c_gop = c_gop.permute(0, 2, 1, 3, 4)

            # select correct frames to compare
            if "PFrame" in self.model.name:
                # remove reference frame
                r_gop = r_gop[:, 1:]
            elif "BFrame" in self.model.name:
                # remove reference frames
                r_gop = r_gop[:, 1: -1]

            # back to CPU
            r_gop = r_gop.cpu()
            c_gop = c_gop.cpu()

        return c_gop, r_gop

    def save_comp_frames(self, dataset="valid"):
        # load nxt GOP
        gop = iter(self.vid_dls[dataset]).next()
        # compress GOP
        c_gop = self._predict_frames(gop)

        # save reference & compressed frames
        vid_t.save_clip("r_clip.mp4", gop[0])
        vid_t.save_clip("c_clip.mp4", c_gop[0])

        return

