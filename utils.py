import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

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
    return parser.parse_args()

def plot_training_curve(args):
    pass

def visualize(args):
    pass