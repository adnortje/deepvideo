# imports
import cv2
import torch


"""
Dense Optical Flow Estimation

    Ref:
        http://www.diva-portal.org/smash/get/diva2:273847/FULLTEXT01.pdf
        optical flow +-= prev[x + dx, y + dy]
"""


class DenseFlow(object):

    def __init__(self):
        self.name = "DenseFlow"

    def __call__(self, video):

        flow_list = []

        for i in range(video.shape[0]-1):

            # luma component
            prvs = cv2.cvtColor(video[i], cv2.COLOR_BGR2GRAY)
            nxt = cv2.cvtColor(video[i+1], cv2.COLOR_BGR2GRAY)

            # calculate Farneback flow
            flow = cv2.calcOpticalFlowFarneback(prvs, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            # append flow
            flow_list.append(
                torch.from_numpy(flow.transpose(2, 0, 1))
            )

        # stack flow vectors
        flow = torch.stack(flow_list, dim=0)

        return flow
