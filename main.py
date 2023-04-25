import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import ConcatDataset

from utils import *
from models import *

def train(args):
    pass

if __name__ == "__main__":
    args = arg_parse()
    print(args)

    if args.train:
        train(args)