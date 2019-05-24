import os
print(os.getcwd())
import SceneR2
from SceneR2.yolov3 import VideoPipeline

pip = VideoPipeline(img_size=416)
pip.vid2vid('data/Daimler/100_vids/videos/20170207_061253_Video.mp4', "tmp")