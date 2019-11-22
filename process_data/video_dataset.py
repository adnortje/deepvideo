# imports
import os
import glob
import skvideo.io as sk
from torch.utils.data.dataset import Dataset

"""
Basic VideoDataset:
    
    Facilitates the creation of an iterable dataset from a folder of Video Clips.
    
        Args:
            root_dir  (string)   : path to directory containing video clip files
            vid_ext   (string)   : video file extension, default is '.mp4'
            transform (callable) : optional transforms to be applied to video frames
    
    Note:
        This is super inefficient as the whole video clip has to be read in then augmented
        (NB: use for Clips < 10 sec)
"""


class VideoDataset(Dataset):

    def __init__(self, root_dir, vid_ext='.mp4', transform=None):

        root_dir = os.path.expanduser(root_dir)

        if not os.path.isdir(root_dir):
            # check directory exists
            raise NotADirectoryError
        else:
            self.root_dir = root_dir

        # extract video file names
        self.video_file_names = glob.glob(
            root_dir+'/*'+vid_ext
        )

        # ensure directory is non-empty
        if len(self.video_file_names) <= 0:
            raise ValueError("Empty Dataset!")

        # transform
        self.transform = transform
        
    def __getitem__(self, index):

        # read video frames
        video_frames = self._get_frames(index)

        if self.transform is not None:
            # apply transform
            video_frames = self.transform(video_frames)

        return video_frames
    
    def _get_frames(self, index):
        # return frames (t, H, W ,C)

        # numpy video data
        video_frames = sk.vread(
            self.video_file_names[index]
        )

        return video_frames
        
    def __len__(self):
        return len(self.video_file_names)

