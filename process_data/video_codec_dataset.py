# imports
import os
import glob
from torch.utils.data.dataset import Dataset

"""
VideoCodecDataset:
    
    Facilitates the creation of an iterable dataset from a folder of Video Clips for standard video codecs.
    
        Args:
            root_dir  (string)   : path to directory containing videos
            vid_ext   (string)   : video file extension, default is mp4
        
        Returns:
            iterator over video file names
"""


class VideoCodecDataset(Dataset):
    
    def __init__(self, root_dir, vid_ext='mp4'):
        
        # check directory exists
        root_dir = os.path.expanduser(root_dir)

        if os.path.isdir(root_dir):
            self.root_dir = root_dir
        else:
            raise NotADirectoryError("Specified directory d.n.e!")

        # get video names
        self.videos = glob.glob(
            self.root_dir+'/*.'+vid_ext
        )
        assert(len(self.videos)>0), "Empty Dataset"
        
    def __getitem__(self, index):
        # return  video path name
        vid = self.videos[index]
        return vid
        
    def __len__(self):
        return len(self.videos)
