import imageio
from PIL import Image
from tqdm import tqdm
import os

vids = [
 '../data/Daimler/100_vids/videos/20170308_082612_Video.mp4',
 '../data/Daimler/100_vids/videos/20170407_223559_Video.mp4']
for vid in tqdm(vids):
    folder = vid.replace('videos', 'images/raw')[:-4]
    print(folder)
    os.makedirs(folder, exist_ok=True)
    reader = imageio.get_reader(vid)
    for i, img in enumerate(tqdm(reader)):
        Image.fromarray(img).save(os.path.join(vid.replace('videos','images/raw')[:-4],str(i).zfill(3)+'.png'))

