# This File contains all the core modules to be imported in all the codes

import torch
from torch import nn as nn
from torch.utils import data as data
import numpy as np
import time


device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Adding layer.freeze and layer.unfreeze capability to nn.Module layers
def freeze(self):
    for param in self.parameters(): param.requires_grad=False
def unfreeze(self):
    for param in self.parameters(): param.requires_grad=True

nn.Module.freeze=freeze
nn.Module.unfreeze=unfreeze