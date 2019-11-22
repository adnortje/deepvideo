# imports
import os
import numpy as np
from warnings import warn
import utils.img_tools as im_t
import matplotlib.pyplot as plt
from adjustText import adjust_text
from sklearn.metrics import auc
from evaluate import EvalVideoModel


"""
Bitrate Loss Curves
    
    Plots Bitrate Loss Curves for a particular model
    
"""


class BitrateLossCurves:

    def __init__(self, models, vdls, rl_lambda, L):
        # model list
        self.models = models

        # video dataLoader
        self.vdls = vdls

        # rate-loss lambda values
        self.rl_lambda = rl_lambda

        # compression levels
        self.L = L

    def create_curve(self, metric, save_loc="./"):

        avg_bpp = []
        avg_met = []

        for i, model in enumerate(self.models, 0):

            # evaluation object
            ev_m = EvalVideoModel(model, self.vdls)

            avg_bpp.append(ev_m.avg_bpp())
            avg_met.append(ev_m.avg_met(metric))

        dict = {"bpp": avg_bpp, "metric": avg_met}

        # save numpy file
        np.save(save_loc + metric + "_" + str(self.L) + ".npy", dict)

        return

    def plot_curve(self, curve_files, title="", savefig=False):

        fmts = ['^--b', '*--c', 'o--g', 'x--r', 'D--r']
        x_points = []
        y_points = []
        l_points = []

        # get metric from path
        fn = os.path.basename(curve_files[0]).split(".")[0]
        metric, _ = fn.split("_")

        # setup plot
        fig = im_t.setup_plot(
            title=title,
            y_label=metric,
            x_label="bits-per-pixel (bpp)",
        )

        ax = fig.add_subplot(111)

        for i, cf in enumerate(curve_files, 0):

            cf = os.path.expanduser(cf)

            if not os.path.isfile(cf):
                raise FileNotFoundError("Specified curve file d.n.e!")

            # get metric from path
            fn = os.path.basename(cf).split(".")[0]
            _, L = fn.split("_")

            # plot curve
            curve = np.load(cf).item()
            plt.plot(curve['bpp'], curve['metric'], fmts[i], label="L="+L, markersize=10)

            if i == 0:
                x_points = x_points + curve['bpp']
                y_points = y_points + curve['metric']
                l_points = l_points + self.rl_lambda
            else:
                x_points = x_points + curve['bpp'][1:]
                y_points = y_points + curve['metric'][1:]
                l_points = l_points + self.rl_lambda[1:]

        xt = curve['bpp'][0]
        yt = curve['metric'][0]

        ax.annotate('Reference Point', xy=(xt, yt-0.2), xytext=(xt-0.001, yt-1.5),
                     arrowprops=dict(facecolor='black', shrink=0.005, connectionstyle="arc3,rad=-0.2"),fontsize="x-large")

        xt = curve['bpp'][2]
        yt = curve['metric'][2]
        ax.annotate('Optimal Point', xy=(xt, yt+0.2), xytext=(xt-0.0015, yt+1.1),
                     arrowprops=dict(facecolor='black', shrink=0.006, connectionstyle="arc3,rad=0.2"),fontsize="x-large")

        texts = [
            plt.text(
                x_points[c],
                y_points[c],
                "\u03BB="+str(l_points[c]),
                ha='center',
                va='center',
                fontsize='medium',
                fontstyle="italic"
            ) for c in range(len(x_points))
        ]
        adjust_text(texts)

        plt.legend(loc="lower right", fontsize="large")
        if savefig:
            plt.savefig("./" + metric + "_" + L + ".pdf")
        plt.show()

        return


"""
Class : CompressionCurves

 Plots compression curves from numpy files

         Args:
            curve_files (list []) : list of numpy compression curve files

"""


class CompressionCurves:

    def __init__(self, curve_files):

        self.curves = []

        for file in curve_files:
            # check files exist
            if os.path.isfile(file):
                self.curves.append(file)
            else:
                warn("Curve file: {}, d.n.e!".format(file))

        if len(self.curves) < 1:
            raise FileNotFoundError("No valid curve files found!")

    def disp_curves(self, save_fig=False):
        # plot compression curves

        # extract metric
        base = os.path.basename(self.curves[0])
        _, metric = os.path.splitext(base)[0].split("_")

        # setup plot
        legend = []
        im_t.setup_plot(title="", y_label=metric, x_label="bits-per-pixel (bpp)")

        # line formats
        fmt_index = 0
        fmts = ['o--g', 'x--r', '^--b', '*--c', 'D--r']

        for curve in self.curves:

            # extract model name & metric
            base = os.path.basename(curve)
            name, met = os.path.splitext(base)[0].split("_")
            legend.append(name)

            if met != metric:
                # sanity check
                raise ValueError("Comparing different metrics!")

            # load curve
            curve = np.load(curve).item()
            plt.plot(curve["bpp"], curve["met"], fmts[fmt_index], linewidth=2, markersize=8)
            plt.xlim((0.15, 0.4))
            fmt_index += 1

        plt.legend(legend, loc="lower right", fontsize="large")

        if save_fig:
            # save plot as pdf
            plt.savefig("./" + metric + "_curve.pdf")

        plt.show()

        return

    def disp_auc(self, bpp_max=2.0, bpp_min=0.0):

        # display Area Under Curve
        print("Displaying AUC:")

        for curve in self.curves:

            # extract model name & metric
            base = os.path.basename(curve)
            name, metric = os.path.splitext(base)[0].split("_")

            # load curve
            curve = np.load(curve).item()

            cut_curve = {"bpp": [], "met": []}

            for i in range(len(curve['bpp'])):

                if bpp_min <= curve['bpp'][i] <= bpp_max:
                    # save values in range
                    cut_curve['bpp'].append(curve['bpp'][i])
                    cut_curve['met'].append(curve['met'][i])

            # calculate & display AUC
            curve_area = auc(cut_curve['bpp'], cut_curve['met'])
            print("{} : {} Curve : {}".format(name, metric, curve_area))

        return
