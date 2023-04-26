import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.models import vgg19

class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = vgg19(pretrained=True).features[:36].eval().to("cuda" if torch.cuda.is_available() else "cpu")
        self.loss = nn.MSELoss()

        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        vgg_input_features = self.vgg(input)
        vgg_target_features = self.vgg(target)
        return self.loss(vgg_input_features, vgg_target_features)

class PSNR:
    def __init__(self, max_val = 1.0):
        self.max_val = max_val

    def __call__(self, img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(self.max_val / torch.sqrt(mse))
    
class MSE:
    def __init__(self):
        pass

    def __call__(self, img1, img2):
        return torch.mean((img1 - img2) ** 2)

class SSIM:
    def __init__(self):
        pass

    def __call__(self, img1, img2):
        pass

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type = str, default = 'results', help = 'Directory to save results')
    parser.add_argument('--data_root', type = str, default = 'data', help = 'Data root directory')
    parser.add_argument('--num_epochs', type = int, default = 30, help = 'Number of epochs to train')
    parser.add_argument('--cuda', action = 'store_true', default = 'cpu', help = 'Device to train on')
    parser.add_argument('--seed', type = int, default = 1999, help = 'Random seed')
    parser.add_argument('--batch_size', type = int, default = 64, help = 'Batch size')
    parser.add_argument('--img_size', type = int, default = 256, help = 'Resizing images for training')
    parser.add_argument('--train', action = 'store_true', help = 'Train the model')
    parser.add_argument('--save_model', action = 'store_true', help = 'Save the model')
    parser.add_argument('--data_aug', action = 'store_true', help = 'Use data augmentation')
    parser.add_argument('--num_workers', type = int, default = 4, help = 'Number of workers for dataloader')
    parser.add_argument('--noise_level', type = float, default = 0.25, help = 'Noise level for training')
    parser.add_argument('--lr', type = float, default = 1e-4, help = 'Learning rate')
    return parser

def plot_training_curve(args):
    pass

def visualize(args):
    pass