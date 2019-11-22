"""
Python Script: 

    downloads a youtube clip given the url

    Requirements:
        youtube-dl
"""
# imports
import os
import pandas as pd
import argparse as arg

# create argument parser
parser = arg.ArgumentParser(
    prog        = 'URL_2_Clips',
    description = 'URL ==> Video Clip'
)
parser.add_argument(
    '-url',
    metavar  = 'URL',
    type     = str,
    required = True,
    help     = 'YouTube URL'
)
parser.add_argument(
    '-od',
    metavar  = 'OutputDir',
    type     = str,
    default = '/home/wintermute/Videos/clip.mp4',
    help     ='Directory to store Video Clips'
)
args = parser.parse_args()

def download_clip(url):
    # shell cmd
    uTube_cmd = ''.join([
        'youtube-dl',
        ' -f mp4',
        ' -o ',
        args.od,
        ' ', 
        url,
        ' --quiet',
        ' --no-warnings'
    ]) 
    # execute shell cmd
    os.system(uTube_cmd)
    return 

# download clip
download_clip(args.url)
    