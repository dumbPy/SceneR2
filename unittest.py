"""This file only unittests the dataset.py
All the classes in dataset.py are tested by the below expressions
"""

from SceneR2.core import *
from SceneR2.dataset import *

data=MovingObjectData.fromCSVFolder("/home/sufiyan/data/Daimler/100_vids/csv/", preload=False)
print(data[0][0])
print(data[0][0].shape)