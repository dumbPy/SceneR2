"""This file only unittests the dataset.py
All the classes in dataset.py are tested by the below expressions
"""
import SceneR2
from SceneR2.core import *
from SceneR2.dataset import *

# data=MovingObjectData2.fromCSVFolder(globalVariables.path_to_csv)
# print(data[0][0])
# print(data[0][0].shape)
# print(data[1])
# print(data[1][1])

with open("/home/sufiyan/data/Daimler/100_vids/pickled_SceneR2_dataset/pickled_MovingObject2_157_NewEdgeDetection_25Feb", 'rb') as f:
    dataset_4=pickle.load(f)

dataset_4.plot(1, all_columns=True, supressPostABA=False)
plt.show()