"""
Url - https://github.com/PyTorchLightning/lightning-bolts/issues/442 # I am trying to implement this feature

In this file.
I am going to create a base or small version of the final project.
----------------------------------------------------------------
Target or What I want to do in this project:
    -
"""

import random
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import torch
import torchvision
from torch.nn import *
from tqdm import tqdm
import cv2
from torch.optim import *
# Model Eval
from sklearn.compose import make_column_transformer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error
# Other
import pickle
import wandb

PROJECT_NAME = 'Weather-archive-Jena-V3'
device = 'cuda:0'
np.random.seed(21)
random.seed(21)
torch.manual_seed(21)
