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

pickled_data=os.path.join(globalVariables.path_to_data, 'Daimler', '100_vids','pickled_SceneR2_dataset', 'pickled_MovObj2_cleaned_157_3Mar')

# globalVariables.path_to_data = 'data/'
# dataset=MovingObjectData2.fromCSVFolder(globalVariables.path_to_csv)
# with open(pickled_data, 'wb') as f:
#     pickle.dump(dataset, f)

globalVariables.video_player = 'cygstart.exe'
with open(pickled_data, 'rb') as f:
    dataset=pickle.load(f)
# dataset.plot(1, all_columns=True, supressPostABA=False)
# plt.show()


dataset.play(1)