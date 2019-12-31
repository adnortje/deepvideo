# Deep Video Compression
## Deep motion estimation for parallel inter-frame prediction in video compression
### Overview
Standard video codecs rely on optical flow to guide inter-frame prediction: 
pixels from reference frames are moved via motion vectors to predict target video frames. 
Here we propose to learn binary motion codes that are encoded based on an input video sequence. 
These codes are not limited to 2D translations, but can capture complex motion (warping, rotation and occlusion). 
Our motion codes are learned as part of a single neural network which also learns to compress and decode them. 
This approach supports parallel video frame decoding instead of the sequential motion estimation 
and compensation of flow-based methods. 
We also introduce 3D dynamic bit assignment to adapt to object displacements caused by motion, 
yielding additional bit savings. 
By replacing the optical flow-based block-motion algorithms found in an existing video codec with our learned inter-frame prediction model, our approach outperforms the standard H.264 and H.265 video codecs at low bitrates.
A preprint of the full length article pertaining to this code is available on arXiv: https://arxiv.org/abs/1912.05193.
### Inter-Frame Motion Compression Sample
The following YouTube link compares our motion compression to that of H.264/5 at a low bitrate.
> https://youtu.be/nV7mLPwOXTI

## Code Status
Please note: This code is currently in a very rough state, 
i.e. it would be hard to use out-of-the-box. 
I'll update and make it more usable in the near future.
This code acts as a good basis for future projects in video compression.

## License
This code is distributed under the Creative Commons Attribution-ShareAlike license.
