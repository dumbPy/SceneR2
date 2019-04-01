import os
from glob import glob
from tqdm import tqdm
import subprocess

vids = glob("../data/Daimler/100_vids/videos/*.mp4")
#print(vids)
for vid in tqdm(vids):
    subprocess.call(f"ffmpeg -i {vid} -filter:v scale=240:-1 -c:a copy {vid.replace('videos', 'videos_180p')}", shell=True)
