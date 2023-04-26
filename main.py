import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import ConcatDataset

from utils import *
from models import *

if __name__ == "__main__":
    args = arg_parse().parse_args(args = "--cuda --num_epochs 5 --train --batch_size 16".split())
    print(args)

    TRAIN_DIR = os.path.join(args.data_root, 'train')
    TEST_DIR = os.path.join(args.data_root, 'test')

    if args.data_aug:
        pass
    else:
        transform = transforms.Compose([
                transforms.Resize((args.img_size, args.img_size)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, ), (0.5, ))
            ])
    train_dataset = ImageFolder(TRAIN_DIR, transform = transform)
    test_dataset = ImageFolder(TEST_DIR, transform = transform)
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, pin_memory = True, num_workers = args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = True)

    if args.cuda:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    generator = Generator(
            in_channels = 1,
            num_channels = 32,
            num_blocks = 6,
            upsample = False
        ).to(device)
    discriminator = Discriminator(
            in_channels = 1
        ).to(device)
    
    if args.train:

        generator.train()
        discriminator.train()
        psnr_train = {}
        psnr_train_denoised = {}
        mse = nn.MSELoss()
        bce = nn.BCEWithLogitsLoss()
        # vgg_loss = VGGLoss()
        gen_optimizer = optim.Adam(generator.parameters(), lr = args.lr, betas = (0.9, 0.999))
        disc_optimizer = optim.Adam(discriminator.parameters(), lr = args.lr, betas = (0.9, 0.999))

        for epoch in range(args.num_epochs):
            loop = tqdm(train_loader, leave = True)
            for idx, (img, _) in enumerate(loop):
                ground_truth = torch.clone(img).to(device)

                img = img.to(device)
                img = img + (args.noise_level * torch.normal(0, 1, img.shape)).to(device)
                gen_denoised = generator(img)
                disc_truth = discriminator(ground_truth)
                disc_denoised = discriminator(gen_denoised.detach())
                
                disc_loss_truth = bce(disc_truth, torch.ones_like(disc_truth) - 0.1 * torch.rand_like(disc_truth))
                disc_loss_denoised = bce(disc_denoised, torch.zeros_like(disc_denoised))
                disc_loss = disc_loss_truth + disc_loss_denoised

                disc_optimizer.zero_grad()
                disc_loss.backward()
                disc_optimizer.step()

                disc_denoised = discriminator(gen_denoised)
                l2_loss = mse(gen_denoised, ground_truth)
                adversarial_loss = 1e-3 * bce(disc_denoised, torch.ones_like(disc_denoised))
                # denoising_vgg_loss = vgg_loss(gen_denoised, ground_truth)
                gen_loss = l2_loss + adversarial_loss #+ denoising_vgg_loss

                gen_optimizer.zero_grad()
                gen_loss.backward()
                gen_optimizer.step()

                loop.set_description(f"Epoch [{epoch + 1}/{args.num_epochs}]")
                loop.set_postfix(gen_loss = gen_loss.item(), disc_loss = disc_loss.item(), l2_loss = l2_loss.item(), adversarial_loss = adversarial_loss.item())

                psnr_train[epoch] = np.mean(PSNR(max_val = 1)(ground_truth, img).detach().cpu().numpy())
                psnr_train_denoised[epoch] = np.mean(PSNR(max_val = 1)(ground_truth, gen_denoised).detach().cpu().numpy())