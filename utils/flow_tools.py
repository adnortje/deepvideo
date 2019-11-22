# imports
import cv2
import numpy as np


"""
Color Code Optical Flow using HSV color space

    Ref:
        https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html

"""


def flow_to_image(flow, fp=None):

    # flow tensor -> numpy
    np_flow = flow.astype(np.float32)

    # u, v vectors
    v, u = np_flow
    # cartesian -> polar coordinates
    mag, ang = cv2.cartToPolar(u, v, angleInDegrees=True)
    # normalize magnitude [0, 100]
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    # hsv image
    hsv = np.zeros((3, flow.shape[1], flow.shape[2]))
    # hue = ang, %saturation = mag, %brightness = 100%
    hsv[0] = ang / 2
    hsv[1] = mag
    hsv[2] = 255
    hsv = hsv.transpose(1, 2, 0)

    # HSV -> BGR
    bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    if fp is not None:
        # write image file
        cv2.imwrite(fp, bgr)
    else:
        # change ordering for PIL compatibility
        rgb = np.flip(bgr, axis=2)

    rgb = rgb.copy().transpose(2, 0, 1)

    return rgb
