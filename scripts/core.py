# This File contains all the core modules to be imported in all the codes

import torch
from torch import nn
import torch.utils.data as data
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import time
from typing import Union
import torchvision
try: from torchnet.meter import ConfusionMeter
except: pass
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm
import pandas as pd
import os, sys, matplotlib.pyplot as plt

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Adding layer.freeze and layer.unfreeze capability to nn.Module layers
def freeze(self):
    for param in self.parameters(): param.requires_grad=False
def unfreeze(self):
    for param in self.parameters(): param.requires_grad=True

nn.Module.freeze=freeze
nn.Module.unfreeze=unfreeze

class oneHot():
    def __init__(self, classes):
        self.c=classes
        self.hot=LabelBinarizer()
        self.hot.fit(range(self.c))

    def __call__(self, y):
        shape_old=np.asarray(y.shape)
        if len(shape_old)>1: y=y.view(shape_old.prod())
        y=y.cpu().numpy()
        y=self.hot.transform(y)
        shape_new=list(shape_old)[:-1]+list(y.shape[-2:])
        return torch.from_numpy(y).float().view(list(shape_new)).to(device)

def read_csv_auto(name):
    try:
        df=pd.read_csv(name)
        if not "ABA_typ_WorkFlowState" in df.columns: raise AttributeError
    except: df=pd.read_csv(name, delimiter=';')
    return df