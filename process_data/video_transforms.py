# imports
import nvvl
import random
from PIL import Image
from torch.nn.modules.utils import _pair

"""
NvvlProcessing
    
    takes input specifications and creates nvvl video processing dictionary
    
    Args:
        frame_size  (int)     : size video frames are to be scaled to (default (0,0) keep original size).
        crop_size   (int)     : size to crop video frame (default (0,0) crops entire frame)
        normalized  (boolean) : set to True to normalize frames to [0, 1], set to False to maintain [0, 255]
        random_crop (boolean) : set to True to crop frame at a random location
        random_flip (boolean) : set to True to randomly flip frames in video sequence.
         
"""


class NvvlProcessing(object):

    def __init__(self, frame_size=(0, 0), crop_size=(0, 0), normalized=True, random_crop=False, random_flip=False,
                 color_space='YCbCr'):

        # frame size
        self.frame_size = _pair(frame_size)
        self.frame_height, self.frame_width = frame_size

        # crop size
        self.crop_size = _pair(crop_size)
        self.crop_height, self.crop_width = crop_size

        # colour space
        if color_space not in ['RGB', 'YCbCr']:
            raise ValueError('Specified Color Space is not supported by NVVL')
        self.color_space = color_space

        # processing dictionary
        self.dict = {
            'input': nvvl.ProcessDesc(
                width=self.crop_width,
                height=self.crop_height,
                scale_width=self.frame_width,
                scale_height=self.frame_height,
                normalized=normalized,
                random_crop=random_crop,
                random_flip=random_flip,
                color_space=color_space
            )
        }

    def get_dictionary(self):
        return self.dict


"""
FirstClipSample:

    samples first GOP from a video

    Args:
        gop_size  (int) : number of consecutive frames to sample

"""


class FirstClipSample(object):

    def __init__(self, gop_size):
        # def no. GOP
        self.gop_size = gop_size

    def __call__(self, video):

        # sanity check
        assert (self.gop_size < len(video)), "GOP size > video!"

        frames = video[0: self.gop_size]

        return frames

    def __repr__(self):
        return self.__class__.__name__ + '(GOP={0})'.format(self.gop_size)


"""
RandomVideoSample:

    randomly samples a GOP from a video
    
    Args:
        gop_size  (int) : number of consecutive frames to sample
        
"""


class RandomVideoSample(object):
    
    def __init__(self, gop_size):

        # def no. GOP
        self.gop_size = gop_size
        
    def __call__(self, video):

        # sanity check
        assert(self.gop_size < len(video)), "GOP size > video!"

        # sample random GOP
        i = random.randint(
            0, (len(video) - self.gop_size) // self.gop_size
        )*self.gop_size

        frames = video[i: i + self.gop_size]
        
        return frames

    def __repr__(self):
        return self.__class__.__name__ + '(GOP={0})'.format(self.gop_size)


"""

 NpFrame2PIL
 
     converts numpy video frame data to PIL image

"""


class NpFrame2PIL(object):

    def __init__(self, color_space):
        # def no. GOP
        self.color_space = color_space

    def __call__(self, frame):
        # convert each frame to PIL Image
        frame = Image.fromarray(frame).convert(self.color_space)
        return frame

    def __repr__(self):
        return self.__class__.__name__ + '(Color Space={0})'.format(self.color_space)


"""

 CropVideoSequence

     cuts video after specified number of frames

 """


class CropVideoSequence(object):

    def __init__(self, num_frames):
        # def no. video frames
        self.num_frames = num_frames

    def __call__(self, video):

        # cut video at specified number of frames

        video = video[0:self.num_frames]

        return video

    def __repr__(self):
        return self.__class__.__name__ + '(Number Frames={0})'.format(self.num_frames)
