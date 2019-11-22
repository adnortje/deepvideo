"""

Script to allow for .avi -> .mp4 transcoding to mitigate the following warnings:

    Warning 1:
        Video uses a non-standard and wasteful way to store B-frames ('packed B-frames').
        Consider using the mpeg4_unpack_bframes bitstream filter without encoding but stream copy to fix it.
        (.avi packs chunks of random B-frames together)

    Warning 2:
        [mp4 @ 0x413c2c0]
        Timestamps are unset in a packet for stream 0. This is deprecated and will stop working in the future.
        Fix your code to set the timestamps properly
        (.avi does not set timestamps for each frame required by .mp4 decoder)

    Note: deletes original .avi file

"""

# imports
import os
import glob
import argparse as arg

# create argument parser
parser = arg.ArgumentParser(
    prog='.avi2.mp4',
    description='filters .avi -> .mp4'
)

parser.add_argument(
    '-vd',
    metavar='VIDEO_DIR',
    type=str,
    required=True,
    help='original .avi video directory.'
)

args = parser.parse_args()

vd = os.path.expanduser(args.vd)

if not os.path.isdir(vd):
    raise NotADirectoryError('Specified input directory d.n.e!')

video_file_names = glob.glob(vd+'/*.mp4')

for i, video in enumerate(video_file_names, 0):

    ffmpeg_cmd = " ".join([
        "ffmpeg",
        "-fflags +genpts",
        "-i",
        "'"+video+"'",
        "-codec copy",
        "-bsf:v mpeg4_unpack_bframes",
        "-an",
        vd+"/"+str(i)+".mp4"
    ])

    # .avi -> .mp4
    os.system(ffmpeg_cmd)

    # remove .avi
    cmd = "rm " + "'"+video+"'"
    os.system(cmd)


