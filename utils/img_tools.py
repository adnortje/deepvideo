# imports
import numpy as np
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

"""
Function: imsave
    saves image in lossless PNG format
"""


def imsave(img, save_loc):

    # place image on CPU
    img = img.cpu()

    if not isinstance(img, np.ndarray):
        # torch.Tensor -> np array
        img = img.numpy()
        np_img = ((np.transpose(img, (1, 2, 0))) * 255).astype(np.uint8)
    else:
        np_img = img

    # save image
    img = Image.fromarray(np_img, mode="RGB")
    img.save(save_loc, format="PNG")
    return


"""
Function: heatshow

    displays a tensor heatmap of an image

        Args:
            heat-map (torch.Tensor) : heat-map tensor (H, W)
"""


def heatshow(heat):

    # place heat on CPU
    heat = heat.cpu()

    if not isinstance(heat, np.ndarray):
        # torch.Tensor -> np array
        np_heat = heat.numpy()
    else:
        np_heat = heat

    # plot img
    plt.figure()
    sns.heatmap(np_heat, vmin=0.0, vmax=1.0, cbar=True, xticklabels=False, yticklabels=False)
    plt.show()

    return


"""
Function: imshow
    
    displays a tensor image
    
        Args:
            img (torch.Tensor) : image tensor (C, H, W)
"""


def imshow(img):

    # place image on CPU
    img = img.cpu()

    if not isinstance(img, np.ndarray):
        # torch.Tensor -> np array
        img = img.numpy()
        np_img = np.transpose(img, (1, 2, 0))
    else:
        np_img = img

    # plot img
    plt.xticks([])
    plt.yticks([])
    plt.imshow(np_img)

    return


"""
Function: setup_plot
    
    creates a new labeled plot
    
    ref: Systems & Signals 414 Course Practical Template (Stellenbosch University)
"""


def setup_plot(title, y_label='', x_label='', newfig=True):

    if newfig:
        fig = plt.figure(figsize=(20, 5))

    plt.title(title, fontdict={"fontsize": "xx-large"})
    plt.ylabel(y_label, fontsize=18)
    plt.xlabel(x_label, fontsize=18)

    return fig


"""
Function: display images

    displays a batch of images

        Args:
            images (torch.Tensor) : batch of images (B, C, H, W)
"""


def display_images(images, nrow=5, padding=2, normalize=False, pad_value=0):
    # display images

    img_grid = make_grid(
        images,
        nrow=nrow,
        padding=padding,
        pad_value=pad_value,
        normalize=normalize
    )

    imshow(img_grid)

    return