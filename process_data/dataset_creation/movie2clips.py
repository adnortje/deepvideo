"""
Python Script: 

used to create a Video Clip Dataset given a full length movie

Requirements:
    ffmpeg
    python3
"""

# imports
import os
import argparse as arg

# create argument parser
parser = arg.ArgumentParser(
    prog='Movie_2_Clips:',
    description='Movie --> Video Clip Dataset'
)

parser.add_argument(
    '--movie_name',
    '-mn',
    metavar='MovieName',
    type=str,
    required=True,
    help='Movie File'
)
parser.add_argument(
    '--n_clips',
    '-nc',
    metavar='NumClips',
    type=int,
    required=True,
    help='No. of Video CLips'
)
parser.add_argument(
    '--c_time',
    '-ct', 
    metavar='ClipTime',
    type=float,
    default=10,
    help='Length of Video Clips (sec)'
)
parser.add_argument(
    '--t_offset',
    '-to',
    metavar='OffsetTime',
    type=int,
    default=0,
    help='Time Offset for Clip extraction (sec)'
)
parser.add_argument(
    '--opt_dir',
    '-od',
    metavar='OutputDir',
    type=str,
    default='./',
    help='Directory to store Video Clips'
)
args = parser.parse_args()

for i in range(args.n_clips):

    t0 = str((i * args.c_time) + args.t_offset)
    t = str(args.c_time)
    m_opt = args.opt_dir + "waterfall" + str(i) + '.mp4'

    # shell cmd
    cmd = ' '.join([
        'ffmpeg',
        '-ss', t0,
        '-i', args.movie_name,
        '-t', t,
        '-r', '24',
        '-an',
        m_opt,
        '-loglevel panic',
        '-hide_banner'
    ])

    # execute shell cmd
    os.system(cmd)

