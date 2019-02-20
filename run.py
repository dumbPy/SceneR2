from scripts.core import *
from scripts.dataset import *

data=MovingObjectData.fromCSVFolder("/home/sufiyan/data/Daimler/100_vids/csv/", preload=False)
print(data[0][0])
print(data[0][0].shape)