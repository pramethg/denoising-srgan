import os
import json
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
    
def psnr(img1, img2, max_val = 1.0):
    mse = np.mean((img1 - img2) ** 2)
    return 20 * np.log10(max_val / np.sqrt(mse))

def mse(img1, img2):
    return np.mean((img1 - img2) ** 2)

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
    parser.add_argument('--eval', action = 'store_true', help = 'Evaluate the model on test data')
    return parser

def test_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type = str, default = 'test_results', help = 'Directory to save results')
    parser.add_argument('--data_root', type = str, default = 'data', help = 'Data root directory')
    parser.add_argument('--cuda', action = 'store_true', default = 'cpu', help = 'Device to train on')
    parser.add_argument('--seed', type = int, default = 1999, help = 'Random seed')
    parser.add_argument('--batch_size', type = int, default = 8, help = 'Batch size')
    parser.add_argument('--img_size', type = int, default = 512, help = 'Resizing images for training')
    parser.add_argument('--noise_level', type = float, default = 0.25, help = 'Noise level for training')
    parser.add_argument('--model_dir', type = str, default = 'results', help = 'Directory to load model')
    return parser

def plot_stats(file = 'results/psnr_train.json', title = 'PSNR vs Epochs', ylabel = 'PSNR'):
    with open(file, 'r') as f:
        stat = json.load(f)
    with open(file[:-5] + '_denoised.json', 'r') as f:
        stat_denoised = json.load(f)
    stat_arr = [stat[i] for i in stat.keys()]
    stat_arr_denoised = [stat_denoised[i] for i in stat_denoised.keys()]
    plt.figure(figsize = (8, 5))
    plt.plot(np.arange(len(stat_arr)), stat_arr, color = 'blue')
    plt.plot(np.arange(len(stat_arr_denoised)), stat_arr_denoised, color = 'green')
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(['Ground Truth vs Input', 'Ground Truth vs Output'])
    plt.show()

def visualize(input, output, gtruth, index = [0, 0, 0], save_dir = None):
    plt.figure(figsize = (15, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(input[index[0]][index[1], index[2]], cmap = "gray")
    plt.title("Input")
    plt.subplot(1, 3, 2)
    plt.imshow(output[index[0]][index[1], index[2]], cmap = "gray")
    plt.title("Output")
    plt.subplot(1, 3, 3)
    plt.imshow(gtruth[index[0]][index[1], index[2]], cmap = "gray")
    plt.title("Ground Truth")
    plt.show()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, "visualize.png"))