"""
Script: calc_cc:

    used to calculate Video Codec Compression Curve values & save values to .npy files
    
"""

# imports
import sys
sys.path.append("..")
import argparse as arg
from video_codec import VideoCodec

# Argument Parser
parser = arg.ArgumentParser(
    prog="Calc Video Codec Compression Curve Values:",
    description="Calc & Stores Metric vs Bpp values"
)
parser.add_argument(
    "--vid_codec",
    "-vc",
    metavar="VIDEO_CODEC",
    type=str,
    choices=["libx264", "libx265", "libvpx-vp9"],
    required=True,
    help="Video compression codec."
)
parser.add_argument(
    "--metric",
    "-m",
    metavar="METRIC",
    type=str,
    choices=["PSNR", "SSIM", "VMAF"],
    required=True,
    help="Video quality metric."
)
parser.add_argument(
    "--n_gop",
    '-ng',
    metavar="N_GOP",
    type=int,
    required=True,
    help="Group Of Pictures (GOP) size."
)
parser.add_argument(
    "--frame_size",
    '-fs',
    metavar="FRAME_SIZE",
    nargs="+",
    type=int,
    required=True,
    help="Video Frame Size."
)
parser.add_argument(
    "--vid_dir",
    '-vd',
    metavar="VIDEO_DIR",
    type=str,
    required=True,
    help="Directory containing video clips."
)
parser.add_argument(
    "--save_loc",
    "-sl",
    metavar="SAVE_LOC",
    type=str,
    required=True,
    help="Directory to save compression curve in."
)
parser.add_argument(
    "--vid_ext",
    "-ve",
    metavar="VIDEO_EXT",
    type=str,
    default="mp4",
    help="Video extension"
)
args = parser.parse_args()

# create Video Codec instance
vc = VideoCodec(
    codec=args.vid_codec,
    n_gop=args.n_gop,
    f_s=args.frame_size,
    root_dir=args.vid_dir,
    vid_ext=args.vid_ext
)

# create compression curve
vc.create_cc(
    metric=args.metric,
    save_loc=args.save_loc
)

# END
