"""
Python Script: 

    used to create a Video Clip Dataset given a csv file containing urls

    Requirements:
        youtube-dl
"""
# imports
import os
import csv
import argparse as arg

# create argument parser
parser = arg.ArgumentParser(
    prog        = 'URL_2_Clips',
    description = 'CSV of URLs ==> Video Clip Dataset'
)
parser.add_argument(
    '-csv',
    metavar  = 'Csv',
    type     = str,
    required = True,
    help     = 'CSV File'
)
parser.add_argument(
    '-od',
    '--opt_dir',
    metavar  = 'OutputDir',
    type     = str,
    default = './',
    help     ='Directory to store Video Clips'
)
parser.add_argument(
    '-nc',
    '--n_clips',
    metavar  = 'NumClips',
    type     = int,
    required = True,
    help     = 'Number of clips to download'
)
args = parser.parse_args()



def download_clip(url, t0, tf, i):
    # shell cmd
    uTube_cmd = ''.join([
        'youtube-dl',
        ' -f mp4 ',
        '--get-url ',
        url,
        ' --quiet',
        ' --no-warnings'
    ]) 
    
    direc = args.opt_dir + 'vid' + str(i) + '.mp4'
    
    ffmpeg_cmd = ''.join([
        'ffmpeg',
        ' -ss ', t0,
        ' -i $(',
        uTube_cmd,
        ')',
        ' -t ', tf,
        ' -c copy',
        ' -an ',
        direc,
        ' -loglevel panic'
    ])
    # execute shell cmd
    os.system(ffmpeg_cmd)
    return 

url_base ='https://www.youtube.com/watch?v='
# read in CSV file
with open(args.csv, "r", newline="") as csvfile:
    reader = csv.reader(csvfile)   
    for i, row in enumerate(reader,0):
        
        y_id = row[0][0:11]
        t0   = row[0][12:18]
        tf   = row[0][19:25]

        y_link = url_base + y_id
        
        # download clip
        download_clip(
            y_link,
            t0,
            tf,
            i
        )
        
        if i == args.n_clips:
            break
    