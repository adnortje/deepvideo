# imports 
import os
import nvvl
import torch
from utils import vid_t
import torchvision.transforms as tf
from .video_dataset import VideoDataset
from torch.utils.data import DataLoader
from torch.nn.modules.utils import _pair
from .nvvl_video_dataset import NvvlVideoDataset
from .video_transforms import RandomVideoSample, NvvlProcessing, NpFrame2PIL, FirstClipSample
        
"""
VideoDataLoaders:
     
     Args:
         f_s      (int)    : video frame size
         root_dir (string) : path to directory containing train, test & validation video sets
         n_gop    (int)    : number of frames that constitute a Group Of Pictures (GOP).
         vid_ext  (string) : video file extension, default .mp4
         b_s      (int)    : batch size
         nvvl     (bool)   : True to use NVIDIA Video Loader
    
     Return:
        Dictionary of video dataLoaders ['train', 'valid', 'test']
"""


class VideoDataLoaders:
    
    def __init__(self, b_s, f_s, n_gop, nvvl=True, color_space="RGB", vid_ext=".mp4", root_dir=None):

        # video frame size
        self.f_s = _pair(f_s)

        # batch size
        self.b_s = b_s

        # frames in GOP
        self.n_gop = n_gop

        # video extension
        self.vid_ext = vid_ext

        # color space
        if color_space not in ['RGB', 'YCbCr']:
            raise ValueError("Specified color_space is not supported!")

        self.color_space = color_space

        # video data directory
        if root_dir is None:
            root_dir = "~/Videos/hollywood/"

        self.root_dir = os.path.expanduser(root_dir)

        # paths to train, valid & test directories
        self.data_paths = {
            "train": self.root_dir + "/train",
            "valid": self.root_dir + "/valid",
            "test": self.root_dir + "/test"
        }

        # NVIDIA Video Loader flag
        self.nvvl = nvvl

    def get_data_loaders(self):
        # dictionary dataLoaders

        if self.nvvl:
            dl = self._get_nvvl_dl()
        else:
            dl = self._get_basic_dl()

        return dl

    def _get_nvvl_dl(self):
        # get nvvl video dataLoaders

        # transform
        video_processing = NvvlProcessing(
            frame_size=self.f_s,
            normalized=True,
            color_space=self.color_space
        ).get_dictionary()

        # train set
        train_set = NvvlVideoDataset(
            root_dir=self.data_paths['train'],
            gop_size=self.n_gop,
            vid_ext=self.vid_ext,
            processing=video_processing
        ).create()

        # train dataLoader
        train_dl = nvvl.VideoLoader(
            train_set,
            batch_size=self.b_s,
            shuffle=True
        )

        # valid dataset
        valid_set = NvvlVideoDataset(
            root_dir=self.data_paths['valid'],
            gop_size=self.n_gop,
            vid_ext=self.vid_ext,
            processing=video_processing
        ).create()

        # valid dataLoader
        valid_dl = nvvl.VideoLoader(
            valid_set,
            batch_size=self.b_s,
            shuffle=True
        )

        # test dataset
        test_set = NvvlVideoDataset(
            root_dir=self.data_paths['test'],
            gop_size=self.n_gop,
            vid_ext=self.vid_ext,
            processing=video_processing
        ).create()

        # test dataLoader
        test_dl = nvvl.VideoLoader(
            test_set,
            batch_size=self.b_s,
            shuffle=False
        )

        # def dataLoader dictionary
        dl = {
            'train': train_dl,
            'valid': valid_dl,
            'test': test_dl
        }

        return dl

    def _get_basic_dl(self):
        # get basic video dataLoaders (inefficient)

        # video frame transform
        frame_transform = tf.Compose([
            NpFrame2PIL(
                self.color_space
            ),
            tf.Resize(self.f_s),
            tf.ToTensor()
        ])

        # train video sequence transform
        train_video_transform = tf.Compose([
            RandomVideoSample(
                self.n_gop
            ),
            tf.Lambda(
                lambda frames: torch.stack(
                    [frame_transform(frame) for frame in frames]
                )
            )
        ])

        # valid video sequence transform
        valid_video_transform = tf.Compose([
            FirstClipSample(
                self.n_gop
            ),
            tf.Lambda(
                lambda frames: torch.stack(
                    [frame_transform(frame) for frame in frames]
                )
            )
        ])

        # train dataset
        train_set = VideoDataset(
            root_dir=self.data_paths['train'],
            vid_ext=self.vid_ext,
            transform=train_video_transform
        )

        # train dataLoader
        train_dl = DataLoader(
            dataset=train_set,
            batch_size=self.b_s,
            shuffle=True,
            num_workers=4
        )
        
        # valid dataset
        valid_set = VideoDataset(
                root_dir=self.data_paths['valid'],
                vid_ext=self.vid_ext,
                transform=valid_video_transform
        )

        # valid dataLoader
        valid_dl = DataLoader(
            dataset=valid_set,
            batch_size=self.b_s,
            shuffle=False,
            num_workers=4
        )
        
        # test dataset
        test_set = VideoDataset(
                root_dir=self.data_paths['test'],
                vid_ext=self.vid_ext,
                transform=valid_video_transform
        )

        # test dataLoader
        test_dl = DataLoader(
            dataset=test_set,
            batch_size=self.b_s,
            shuffle=False,
            num_workers=2
        )
        
        # def dataLoader Dictionary
        dl = {
            'train': train_dl,
            'valid': valid_dl,
            'test': test_dl
        }
        
        return dl
    
    def display_gop(self, dataset, widget=False):
        # displays GOP alongside one another

        # create dataLoader
        dl = self._get_basic_dl()[dataset]

        # fetch video sequence
        gop = iter(dl).next()[0]

        # display GOP
        if widget:
            vid_t.display_frames_widget(gop)
        else:
            vid_t.display_frames(gop)

        return
    
    def play_gop(self, dataset):
        # play GOP clip

        # create dataLoader
        dl = self._get_basic_dl()[dataset]

        # video frames
        gop = iter(dl).next()[0]

        # play GOP clip
        vid_t.play_clip(gop)

        return
