"""
Python Script used to transcode video files to mp4

"""

# imports
import os
import glob
import argparse as arg

# create argument parser
parser = arg.ArgumentParser(
    prog='Clips2NvvL',
    description='Video Clips -> Clips supported by Nvvl'
)

parser.add_argument(
    '-vd',
    metavar='VIDEO_DIR',
    type=str,
    required=True,
    help='Original video directory.'
)

parser.add_argument(
    '-od',
    metavar='OUTPUT_DIR',
    type=str,
    default='./',
    help='Directory to store Nvvl video clips'
)


parser.add_argument(
    '-ext',
    metavar='VID_EXT',
    type=str,
    default='.mp4',
    help='Video file extensions'
)

args = parser.parse_args()

od = os.path.expanduser(args.od)

if not os.path.isdir(od):
    raise NotADirectoryError('Specified output directory d.n.e!')

vd = os.path.expanduser(args.vd)

if not os.path.isdir(vd):
    raise NotADirectoryError('Specified input directory d.n.e!')

video_file_names = glob.glob(vd+'/*'+args.ext)

for i, video_file in enumerate(video_file_names, 0):

    # transcoded file name
    output_file = od + '/' + str(i) + '.mp4'

    print("ENCODING -> " + str(i) + ".mp4\n")
    # ffmpeg command
    cmd = " ".join([
        "ffmpeg",
        "-i",
        "'"+video_file+"'",
        "-c:v libx264",
        "-r 24",
        "-an",
        "-profile:v high",
        output_file
    ])

    # run command
    os.system(cmd)
    print("")
