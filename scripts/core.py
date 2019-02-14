# This File contains all the core modules to be imported in all the codes

import torch
from torch import nn
import torch.utils.data as data
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np, math
import time
from typing import Union
import torchvision
try: from torchnet.meter import ConfusionMeter
except: pass
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm, tqdm_notebook
import pandas as pd
import os, sys, matplotlib.pyplot as plt
from scipy.ndimage.filters import laplace, gaussian_filter1d
from functools import partial
from pathlib import Path

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
        """One Hot Encoder for torch.Tensors
        en=oneHot(num_of_classes) to initialize the OneHot Encoder
        then en(tensorToOneHotEncode) to get one hot encoded version of it.
        """
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

def read_csv_auto(filename):
    """pandas.read_csv wrapper that deals with both delimiters (, and ;)
    """
    try:
        df=pd.read_csv(filename)
        # if not "ABA_typ_WorkFlowState" in df.columns: raise AttributeError
        #all cols read into 1 column due to different delimiter 
        if len(df.columns)==1: raise AttributeError 
    except: df=pd.read_csv(filename, delimiter=';')
    return df

#defines all the columns we are interested in
allCols=["ABA_typ_WorkFlowState", "OPC_typ_BrakeReq", "ABA_typ_ABAAudioWarn", "ABA_typ_SelObj",
        "BS_v_EgoFAxleLeft_kmh", "BS_v_EgoFAxleRight_kmh", "RDF_val_YawRate", "RDF_typ_ObjTypeOr",
        "RDF_dx_Or", "RDF_v_RelOr", "RDF_dy_Or", "RDF_typ_ObjTypeOs", "RDF_dx_Os", "RDF_v_RelOs",
        "RDF_dy_Os", "RDF_typ_ObjTypePed0", "RDF_dx_Ped0", "RDF_vx_RelPed0", "RDF_dy_Ped0"]
