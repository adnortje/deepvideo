# imports
import os
from utils import img_t
import torchvision.transforms as tf
from .image_dataset import ImageDataset
from torch.utils.data import DataLoader
from torch.nn.modules.utils import _pair


"""
ImgDataLoaders 
    
    Args:
        b_s   (int)        : batch size
        img_s (int, tuple) : image size
        root_dir (string)  : directory with train, test & validation image sets
        
    Returns:
        dictionary of dataLoaders [train, valid, test]
"""


class ImgDataLoaders:
    
    def __init__(self, b_s, img_s, root_dir=None):

        # batch size
        self.b_s = b_s

        # image size
        self.img_s = _pair(img_s)

        # train, valid, test root directory
        if root_dir is not None:
            self.root_dir = os.path.expanduser(root_dir)
        else:
            self.root_dir = os.path.expanduser(
                '~/Pictures/Clic/Professional/'
            )

        self.data_rt = {
            'train': self.root_dir + 'train',
            'valid': self.root_dir + 'valid',
            'test': self.root_dir + 'test'
        }
        
    def get_data_loaders(self):
        # return dictionary of train, valid & test image dataLoaders
        
        # def tfs
        img_transform = tf.Compose([
            tf.Resize(self.img_s),
            tf.ToTensor()
        ])

        # train set
        train_set = ImageDataset(
            root_dir=self.data_rt['train'],
            transform=img_transform
        )

        # train data loader
        train_dl = DataLoader(
            dataset=train_set,
            batch_size=self.b_s,
            shuffle=True,
            num_workers=4
        )
        
        # valid set
        valid_set = ImageDataset(
            root_dir=self.data_rt['valid'],
            transform=img_transform
        )

        # valid dl
        valid_dl = DataLoader(
            dataset=valid_set,
            batch_size=self.b_s,
            shuffle=True,
            num_workers=4
        )
        
        # test set
        test_set = ImageDataset(
            root_dir=self.data_rt['test'],
            transform=img_transform
        )

        # test dl
        test_dl = DataLoader(
            dataset=test_set,
            batch_size=1,
            shuffle=False,
        )

        # def dl Dict 
        dl = {
            'train': train_dl,
            'valid': valid_dl,
            'test': test_dl
        }
        
        return dl
    
    def display_images(self, dataset):

        # load image batch
        images = iter(self.get_data_loaders()[dataset]).next()

        # display images
        img_t.display_images(images, normalize=True)

        return
