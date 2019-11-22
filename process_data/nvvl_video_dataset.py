# imports
import os
import glob
import nvvl

"""
NvvlVideoDataset

    Facilitates the creation of an iterable dataset from a folder of Video Clips by leveraging NVVL.

    Args:
        root_dir   (string)     : path to directory containing video clip files
        gop_size   (int)        : Group Of Pictures size
        vid_ext    (string)     : video file extension, default is '.mp4'
        processing (Dictionary) : key value pairs of nvvl video processing options 

    Ref:
        https://github.com/NVIDIA/nvvl

    Note: Super efficient implementation (can be used on full length movies). 

"""


class NvvlVideoDataset(object):

    def __init__(self, root_dir, gop_size, vid_ext=".mp4", processing=None):

        root_dir = os.path.expanduser(root_dir)

        if not os.path.isdir(root_dir):
            # check video directory exists
            raise NotADirectoryError
        else:
            self.root_dir = root_dir

        # extract video file names
        self.video_file_names = glob.glob(
            self.root_dir + '/*' + vid_ext
        )

        # check Dataset is non-empty
        if len(self.video_file_names) <= 0:
            raise ValueError("Empty Dataset!")

        # def GOP size
        self.gop_size = gop_size

        # video processing
        self.processing = processing

    def create(self):
        # create nvvl dataset

        video_dataset = nvvl.VideoDataset(
            filenames=self.video_file_names,
            sequence_length=self.gop_size,
            processing=self.processing
        )

        return video_dataset
