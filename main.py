import os
import json
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
    args = arg_parse().parse_args(args = "--cuda --num_epochs 30 --train --batch_size 32 --img_size 128 --noise_level 0.2 --save_model --save_dir results_02_aug --data_aug".split())
    print(args)

    TRAIN_DIR = os.path.join(args.data_root, 'train')
    TEST_DIR = os.path.join(args.data_root, 'test')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.data_aug:
        train_transform_1 = transforms.Compose([
                transforms.Resize((args.img_size, args.img_size)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, ), (0.5, ))
            ])
        train_transform_2 = transforms.Compose([
                transforms.Resize((args.img_size, args.img_size)),
                transforms.Grayscale(),
                transforms.RandomHorizontalFlip(p = 0.5),
                transforms.RandomVerticalFlip(p = 0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.5, ), (0.5, ))
            ])
        train_transform_3 = transforms.Compose([
                transforms.RandomRotation(90),
                transforms.Resize((args.img_size, args.img_size)),
                transforms.Grayscale(),
                transforms.RandomHorizontalFlip(p = 0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.5, ), (0.5, ))
            ])
    else:
        transform = transforms.Compose([
                transforms.Resize((args.img_size, args.img_size)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, ), (0.5, ))
            ])
        
    test_transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, ), (0.5, ))
            ])
    
    if args.data_aug:
        train_dataset_1 = ImageFolder(TRAIN_DIR, transform = train_transform_1)
        train_dataset_2 = ImageFolder(TRAIN_DIR, transform = train_transform_2)
        train_dataset_3 = ImageFolder(TRAIN_DIR, transform = train_transform_3)
        train_dataset = ConcatDataset([train_dataset_1, train_dataset_2, train_dataset_3])
    else:
        train_dataset = ImageFolder(TRAIN_DIR, transform = transform)

    test_dataset = ImageFolder(TEST_DIR, transform = test_transform)
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, pin_memory = True, num_workers = args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size // 4, shuffle = True)

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
        psnr_train, psnr_train_denoised = {}, {}
        mse_train, mse_train_denoised = {}, {}
        mse = nn.MSELoss()
        bce = nn.BCEWithLogitsLoss()
        # vgg_loss = VGGLoss()
        gen_optimizer = optim.Adam(generator.parameters(), lr = args.lr, betas = (0.9, 0.999))
        disc_optimizer = optim.Adam(discriminator.parameters(), lr = args.lr, betas = (0.9, 0.999))

        for epoch in range(args.num_epochs):
            psnr_train_list, psnr_train_denoised_list = [], []
            mse_train_list, mse_train_denoised_list = [], []
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

                psnr_train_list.append((PSNR(max_val = 1)(ground_truth, img).detach().cpu().numpy()).item())
                psnr_train_denoised_list.append((PSNR(max_val = 1)(ground_truth, gen_denoised).detach().cpu().numpy()).item())
                mse_train_list.append(MSE()(ground_truth, img).detach().cpu().numpy().item())
                mse_train_denoised_list.append(MSE()(ground_truth, gen_denoised).detach().cpu().numpy().item())

            psnr_train[epoch] = np.mean(psnr_train_list)
            psnr_train_denoised[epoch] = np.mean(psnr_train_denoised_list)
            mse_train[epoch] = np.mean(mse_train_list)
            mse_train_denoised[epoch] = np.mean(mse_train_denoised_list)

            print(f"Epoch [{epoch + 1}/{args.num_epochs}] PSNR: {psnr_train[epoch]} PSNR Denoised: {psnr_train_denoised[epoch]}")

        if args.save_model:
            torch.save(generator.state_dict(), os.path.join(args.save_dir, 'generator.pth'))
            torch.save(discriminator.state_dict(), os.path.join(args.save_dir, 'discriminator.pth'))

        with open(os.path.join(args.save_dir, 'psnr_train.json'), 'w') as f:
            json.dump(psnr_train, f)

        with open(os.path.join(args.save_dir, 'psnr_train_denoised.json'), 'w') as f:
            json.dump(psnr_train_denoised, f)

        with open(os.path.join(args.save_dir, 'mse_train.json'), 'w') as f:
            json.dump(mse_train, f)

        with open(os.path.join(args.save_dir, 'mse_train_denoised.json'), 'w') as f:
            json.dump(mse_train_denoised, f)
    
    if args.eval:
        generator.load_state_dict(torch.load(os.path.join(args.save_dir, 'generator.pth')))
        generator.eval()

        psnr_test = []
        psnr_test_denoised = []
        mse_test_denoised = []
        inputs, outputs, ground_truths = [], [], []

        loop = tqdm(test_loader, leave = True)
        for idx, (img, _) in enumerate(loop):
            ground_truth = torch.clone(img).to(device)
            img = img.to(device)
            img = img + (args.noise_level * torch.normal(0, 1, img.shape)).to(device)
            gen_denoised = generator(img)
            inputs.append(img.detach().cpu().numpy())
            outputs.append(gen_denoised.detach().cpu().numpy())
            ground_truths.append(ground_truth.detach().cpu().numpy())

            psnr_test.append((PSNR(max_val = 1)(ground_truth, img).detach().cpu().numpy()).item())
            psnr_test_denoised.append((PSNR(max_val = 1)(ground_truth, gen_denoised).detach().cpu().numpy()).item())
            mse_test_denoised.append((MSE()(ground_truth, gen_denoised).detach().cpu().numpy()).item())
            del img, gen_denoised, ground_truth
        print(f"PSNR: {np.mean(psnr_test)} PSNR Denoised: {np.mean(psnr_test_denoised)} MSE Denoised: {np.mean(mse_test_denoised)}")