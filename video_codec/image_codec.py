# imports
import os
import tempfile
from .ffmpeg import FFmpeg
import torchvision.transforms as tf
import skvideo.io as sk

"""
ImageCodec

    Uses:
        encodes image using image codec

        Supported Standard Lossy Codecs:
            - libx265
            - libx264
            Note: should work with other codecs supported by PIL
        
        Supported Deep Codecs:
            - # TODO: add deep image codecs
"""


class ImageCodec:

    def __init__(self, codec):
        self.codec = codec
        self.v_coder = FFmpeg(codec, 1)

    def encode_decode(self, r_img, q):
        # encode & decode with codec

        # save image
        img_tmp = tempfile.NamedTemporaryFile(suffix='.png')
        r_img = tf.ToPILImage()(r_img)
        r_img.save(img_tmp, format="PNG")

        # encode using ffmpeg
        dec_img = self.v_coder.encode_img(img_tmp.name, q)

        # calculate bpp
        w, h = r_img.size
        bits = os.path.getsize(dec_img.name) * 8

        c_img = tf.ToTensor()(sk.vread(dec_img.name)[0])

        # close image buffer
        img_tmp.close()
        dec_img.close()

        return c_img, bits

