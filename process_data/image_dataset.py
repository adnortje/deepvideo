# imports
import os
import glob
from PIL import Image
from torch.utils.data import Dataset

"""
Class: ImageDataset

    Extends the torch Dataset class. Facilitates the creation of an iterable 
    dataset from an image folder.
    
    Args:
        root_dir  (string)             : path to directory containing images
        transform (callable, optional) : optional transforms to be applied to a sample image
        img_ext   (string)             : image file extension, default is png
        
"""


class ImageDataset(Dataset):
    
    def __init__(self, root_dir, transform=None, img_ext='.png'):
        
        # get image names
        root_dir = os.path.expanduser(root_dir)

        if not os.path.isdir(root_dir):
            raise NotADirectoryError

        self.img_names = glob.glob(
            root_dir+'/*'+img_ext
        )

        if not len(self.img_names) > 0:
            raise ValueError('Empty Directory')

        # transform
        self.transform = transform

    def __getitem__(self, idx):
        
        # get image file name
        img_name = self.img_names[idx]

        # open image as RGB
        img = Image.open(
            img_name
        ).convert('YCbCr')
        
        # apply transform
        if self.transform is not None:
            img = self.transform(img)

        return img
    
    def __len__(self):
        return len(self.img_names)
