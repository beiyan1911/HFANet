#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
import time
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from skimage.metrics import peak_signal_noise_ratio as ski_psnr
from skimage.metrics import structural_similarity as ski_ssim
from torch.utils.data import DataLoader
import utils
from utils import NpzData
from model import HFANetWork
from utils import Perceptual_vgg_loss

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Hierarchical-feature Fusion Attention Network')
    parser.add_argument('--gpu_num', '-g', type=int, default=1, help='Num. of GPUs')
    parser.add_argument('--output_root', type=str, default='./outputs/', help='output folder')
    parser.add_argument('--num_work', default=0, type=int, help='threads for loading data')
    parser.add_argument('--checkpoints', type=str, default='checkpoints/epoch_1.pth', help='which epoch to load?')
    parser.add_argument('--save_image', default=False, help='whether to save image in valid phase?')
    parser.add_argument('--resume', default=False, help='continue training: True or False')
    parser.add_argument('--resume_count', type=int, default=1, help='when resume,from which count to epoch')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size on train iter')
    parser.add_argument('--epoch_num', default=100, type=int, help='total epoch num for train')
    parser.add_argument('--train_path', type=str, default='datasets/train', help='npz folder for train')
    parser.add_argument('--valid_path', type=str, default='datasets/valid', help='npz folder for valid')
    args = parser.parse_args()
    device = torch.device("cuda" if args.gpu_num > 0 else "cpu")

    init_time = time.time()

    train_loader = DataLoader(dataset=NpzData(args.train_path), num_workers=args.num_work,
                              batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(dataset=NpzData(args.valid_path), num_workers=0,
                              batch_size=args.batch_size, shuffle=False, pin_memory=False)

    torch.manual_seed(2020)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    # model
    model = HFANetWork().to(device)

    # loss
    L1Loss = nn.L1Loss().to(device)
    perceptualLoss = Perceptual_vgg_loss().to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch_num, eta_min=1e-6)

    # create output folders
    if not os.path.exists(args.output_root):
        os.makedirs(os.path.join(args.output_root, 'valid_out'))
        os.makedirs(os.path.join(args.output_root, 'checkpoints'))

    test_interval = 1
    epoch_start = 1
    # load params from ckpt when it is required
    if args.resume:
        model.load_state_dict(torch.load(args.output_root + args.checkpoints, map_location=device))
        epoch_start = args.resume_count
        scheduler.step(epoch_start - 1)
        print('load ckpt from {} , and restart train from epoch {} '.format(args.checkpoints, epoch_start))

    # Train loop
    for epoch in range(epoch_start, args.epoch_num + 1):
        start_time = time.time()
        print('epoch', epoch)
        train_loss_l1 = 0.0
        train_loss_p = 0.0
        for module in model.children():
            module.train(True)

        for ite, (input, target) in enumerate(train_loader):
            lr_patch = input.to(device)
            hr_patch = target.to(device)
            optimizer.zero_grad()
            output = model(lr_patch)
            l1_loss = L1Loss(output, hr_patch)
            p_loss = 0.0006 * perceptualLoss(output, hr_patch)
            loss = l1_loss + p_loss
            loss.backward()
            optimizer.step()
            train_loss_l1 += l1_loss.item()
            train_loss_p += p_loss.item()

        print('Train set : l1 loss: {:.4f} p loss: {:.4f} '.format(train_loss_l1, train_loss_p))
        print('time ', time.time() - start_time)

        # save ckpt each epoch
        utils.save_networks(model, save_path=os.path.join(args.output_root, 'checkpoints', 'epoch_%d.pth' % epoch))
        scheduler.step(epoch)

        # check val performance
        if epoch % test_interval == 0:
            test_start_time = time.time()
            with torch.no_grad():
                print('------------------------')
                for module in model.children():
                    module.train(False)
                test_ite = 0
                test_psnr = 0
                test_ssim = 0
                eps = 1e-10

                for ti, (input, target) in enumerate(valid_loader):
                    realA = input.to(device)
                    realB = target.to(device)
                    fakeB = model(realA)
                    if args.save_image:
                        vutils.save_image(fakeB,
                                          os.path.join(args.output_root, 'valid_out',
                                                       '%05d_%05d_outputs.png' % (epoch, ti)),
                                          padding=0, normalize=True)
                        vutils.save_image(realA,
                                          os.path.join(args.output_root, 'valid_out',
                                                       '%05d_%05d_inputs.png' % (epoch, ti)),
                                          padding=0, normalize=True)
                        vutils.save_image(realB,
                                          os.path.join(args.output_root, 'valid_out',
                                                       '%05d_%05d_targets.png' % (epoch, ti)),
                                          padding=0, normalize=True)
                    # Calculate SSIM and PSNR
                    fakeB = fakeB.data.cpu().numpy()
                    realB = realB.data.cpu().numpy()
                    fakeB = fakeB.transpose((0, 2, 3, 1))  # N C H W  ---> N H W C
                    realB = realB.transpose((0, 2, 3, 1))  # N C H W  ---> N H W C
                    fakeB = np.clip(fakeB, a_min=0.0, a_max=1.0)
                    realB = np.clip(realB, a_min=0.0, a_max=1.0)

                    for _bti in range(fakeB.shape[0]):
                        per_fakeB = fakeB[_bti]
                        per_realB = realB[_bti]
                        test_ssim += ski_ssim(per_fakeB, per_realB, data_range=1, multichannel=True)
                        test_psnr += ski_psnr(per_realB, per_fakeB, data_range=1)
                        test_ite += 1

                test_psnr /= test_ite
                test_ssim /= test_ite
                print('     Valid PSNR: {:.4f}'.format(test_psnr))
                print('     Valid SSIM: {:.4f}'.format(test_ssim))
                print('------------------------')
