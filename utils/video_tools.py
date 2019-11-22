# imports
import torch
import tempfile
import numpy as np
from utils import img_t
import skvideo.io as sk
from ipywidgets import *
from .flow_tools import flow_to_image
from torchvision.utils import make_grid


"""
Function: display_flow

    displays a sequence of video flow alongside one another

        Args:
            heat-maps (torch.Tensor) : heat-map sequence sequence (T, H, W)
"""


def display_heatmap(heatmaps):

    if isinstance(heatmaps, np.ndarray):
        heatmaps = torch.from_numpy(heatmaps)

    # unsqueeze channel dimension
    heatmaps = heatmaps.unsqueeze(1)

    # plot frames consecutively
    heatmaps = make_grid(heatmaps, nrow=heatmaps.size(0), padding=1, pad_value=1.0)
    img_t.heatshow(heatmaps[0])

    return


"""
Function: display_flow

    displays a sequence of video flow alongside one another

        Args:
            flow (torch.Tensor) : flow sequence (T, C, H, W)
"""


def display_flow(flows):

    if not isinstance(flows, np.ndarray):
        flows = flows.numpy()

    flows = [
        torch.from_numpy(flow_to_image(flow)) for flow in flows
    ]

    flows = torch.stack(flows, dim=0)

    # plot frames consecutively
    flows = make_grid(flows, nrow=flows.size(0), padding=1)
    img_t.imshow(flows)

    return


"""
Function: vs_display_flow

    displays a sequence of compressed and reference flow alongside one another

        Args:
            frames (torch.Tensor) : frame sequence (T, C, H, W)
"""


def vs_display_flow(r_flow, c_flow):

    if r_flow.size(0) != c_flow.size(0):
        # sanity check
        raise ValueError("Unequal flow lengths!")

    if not isinstance(r_flow, np.ndarray):
        r_flow = r_flow.numpy()

    if not isinstance(c_flow, np.ndarray):
        c_flow = c_flow.numpy()

    r_flow = [
        torch.from_numpy(flow_to_image(flow)) for flow in r_flow
    ]

    c_flow = [
        torch.from_numpy(flow_to_image(flow)) for flow in c_flow
    ]

    r_flow = torch.stack(r_flow, dim=0)
    c_flow = torch.stack(c_flow, dim=0)

    # display frames
    flow = torch.cat([r_flow, c_flow], dim=0)
    flow = make_grid(flow, nrow=r_flow.size(0), padding=1)
    img_t.imshow(flow)

    return


"""
Function: vs_display_flow_widget

        displays reference & compressed video frames alongside one another

        Args:
            r_frames (torch.Tensor) : reference flow
            c_frames (torch.Tensor) : compressed flow
"""


def vs_display_flow_widget(r_flow, c_flow):

    if r_flow.size(0) != c_flow.size(0):
        # sanity check
        raise ValueError("Unequal flow lengths!")

    if not isinstance(r_flow, np.ndarray):
        r_flow = r_flow.numpy()

    if not isinstance(c_flow, np.ndarray):
        c_flow = c_flow.numpy()

    r_flow = [
        torch.from_numpy(flow_to_image(flow)) for flow in r_flow
    ]

    c_flow = [
        torch.from_numpy(flow_to_image(flow)) for flow in c_flow
    ]

    def imshow_callback(ref, comp, i):
        # display images alongside one another
        img = make_grid([ref[i], comp[i]], nrow=2, padding=1)
        img_t.imshow(img)
        return

    # display widget
    interact(imshow_callback, ref=fixed(r_flow), comp=fixed(c_flow), i=(0, len(r_flow) - 1, 1))

    return


"""
Function: display_frames
    
    displays a sequence of video frames alongside one another
        
        Args:
            frames (torch.Tensor) : frame sequence (T, C, H, W)
"""


def display_frames(frames):

    if isinstance(frames, np.ndarray):
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)

    # plot frames consecutively
    frames = make_grid(frames, nrow=frames.size(0), padding=2, pad_value=255)
    img_t.imshow(frames)

    return


"""
Function: vs_display_frames

    displays a sequence of compressed and reference video frames alongside one another

        Args:
            frames (torch.Tensor) : frame sequence (T, C, H, W)
"""


def vs_display_frames(r_frames, c_frames):

    if r_frames.size(0) != c_frames.size(0):
        # sanity check
        raise ValueError("Unequal clip lengths!")

    # display frames
    frames = torch.cat([r_frames, c_frames], dim=0)
    img_t.imshow(frames)

    return


"""
Function: display_frames_widget

    displays a frame sequence as an interactive widget

        Args:
            frames (torch.Tensor) : frame sequence (t, c, h, w)
"""


def display_frames_widget(frames):
    def imshow_callback(f, i):
        # display image
        img_t.imsave(f[i], "i_" + str(i) + ".png")
        img_t.imshow(f[i])
        return

    # display frames widget
    interact(imshow_callback, f=fixed(frames), i=(0, frames.size(0) - 1, 1))

    return


"""
Function: vs_display_frames_widget score
        
        displays reference & compressed video frames alongside one another
        
        Args:
            r_frames (torch.Tensor) : reference video frames
            c_frames (torch.Tensor) : compressed and decoded video frames
"""


def vs_display_frames_widget(r_frames, c_frames):

    if r_frames.size(0) != c_frames.size(0):
        # sanity check
        raise ValueError("Unequal clip lengths!")

    def imshow_callback(ref, comp, i):
        # display images alongside one another
        img = make_grid([ref[i], comp[i]], nrow=2, padding=1)
        img_t.imshow(img)
        return

    # display widget
    interact(imshow_callback, ref=fixed(r_frames), comp=fixed(c_frames), i=(0, r_frames.size(0)-1, 1))

    return


"""
Function: save_clip
    
    save frame sequence as video clip
        
        Args:
            file_name (string)       : video file name
            frames    (torch.Tensor) : frame sequence (t, c, h, w)
"""


def save_clip(file_name, frames):
    np_frames = np.transpose(frames.numpy(), (0, 2, 3, 1)) * 255
    np_frames = np_frames.astype(np.uint8)
    sk.vwrite(file_name, np_frames)
    return


"""
Function: play_clip

    plays a sequence of video frames

        Args:
            frames (torch.Tensor) : frame sequence (T, C, H, W)
"""


def play_clip(frames):
    # torch.Tensor -> Numpy
    np_frames = np.transpose(frames.numpy(), (0, 2, 3, 1)) * 255
    np_frames = np_frames.astype(np.uint8)

    # write frames to temp file
    temp_file = tempfile.NamedTemporaryFile(
        suffix='.mp4'
    )
    sk.vwrite(temp_file.name, np_frames)

    # play video clip
    cmd = 'ffplay -i ' + temp_file.name + ' -loop 0'
    os.system(cmd)

    # close & delete temp file
    temp_file.close()

    return
