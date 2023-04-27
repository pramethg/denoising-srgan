import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

from utils import *
from models import *

if __name__ == "__main__":
    args = test_args().parse_args(args = '--noise_level 0.2 --batch_size 8 --seed 1999 --save_dir test_02_noaug'.split())
    print(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    TEST_DIR = os.path.join(args.data_root, 'test')
    test_transform = transforms.Compose([
                transforms.Resize((args.img_size, args.img_size)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, ), (0.5, ))
            ])
    test_dataset = ImageFolder(TEST_DIR, transform = test_transform)
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
    generator.load_state_dict(torch.load(os.path.join(args.model_dir, "generator.pth"), map_location = device))
    generator.eval()
    test_loop = tqdm(test_loader, leave = True)

    psnr_test, psnr_test_denoised = [], []
    mse_test, mse_test_denoised = [], []
    inputs, outputs, ground_truths = [], [], []

    for idx, (img, _) in enumerate(test_loop):
        ground_truth = torch.clone(img).to(device)
        img = img.to(device)
        img = img + (args.noise_level * torch.normal(0, 1, img.shape)).to(device)
        gen_denoised = generator(img)

        inputs.append(img.detach().cpu().numpy())
        outputs.append(gen_denoised.detach().cpu().numpy())
        ground_truths.append(ground_truth.detach().cpu().numpy())
        psnr_test.append((PSNR(max_val = 1)(ground_truth, img).detach().cpu().numpy()).item())
        psnr_test_denoised.append((PSNR(max_val = 1)(ground_truth, gen_denoised).detach().cpu().numpy()).item())
        mse_test.append((MSE()(ground_truth, img).detach().cpu().numpy()).item())
        mse_test_denoised.append((MSE()(ground_truth, gen_denoised).detach().cpu().numpy()).item())
        del img, gen_denoised, ground_truth

    inputs = np.squeeze(np.concatenate(inputs, axis = 0), axis = 1)
    outputs = np.squeeze(np.concatenate(outputs, axis = 0), axis = 1)
    ground_truths = np.squeeze(np.concatenate(ground_truths, axis = 0), axis = 1)

    np.save(os.path.join(args.save_dir, "inputs.npy"), inputs)
    np.save(os.path.join(args.save_dir, "outputs.npy"), outputs)
    np.save(os.path.join(args.save_dir, "ground_truths.npy"), ground_truths)

    for i in range(0, 360, 10):
        visualize(inputs, outputs, ground_truths, i, args.save_dir)

    print(f"PSNR: {np.mean(psnr_test)} PSNR Denoised: {np.mean(psnr_test_denoised)} MSE Test: {np.mean(mse_test)} MSE Denoised: {np.mean(mse_test_denoised)}")