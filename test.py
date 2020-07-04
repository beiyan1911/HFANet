#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
import time
import numpy as np
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.utils as vutils
from skimage.metrics import peak_signal_noise_ratio as ski_psnr
from skimage.metrics import structural_similarity as ski_ssim
from torch.utils.data import DataLoader
from model import HFANetWork
from utils import ImageData

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Hierarchical-feature Fusion Attention Network')
    parser.add_argument('--gpu_num', '-g', type=int, default=1, help='Num. of GPUs')
    parser.add_argument('--output_root', type=str, default='./outputs/', help='output folder')
    parser.add_argument('--num_work', default=0, type=int, help='threads for loading data')
    parser.add_argument('--checkpoints', type=str, default='checkpoints/epoch_1.pth', help='which epoch to load?')
    parser.add_argument('--save_image', default=True, help='whether to save image in valid phase?')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size on train iter')
    parser.add_argument('--test_path', type=str, default='datasets/test', help='npz folder for valid')
    args = parser.parse_args()
    device = torch.device("cuda" if args.gpu_num > 0 else "cpu")

    init_time = time.time()
    test_sets_dict = {'moderate': ImageData(args.test_path, mode='moderate'),
                      'mild': ImageData(args.test_path, mode='mild'),
                      'severe': ImageData(args.test_path, mode='severe')}

    torch.manual_seed(2020)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    # model
    model = HFANetWork().to(device)
    for module in model.children():
        module.train(False)
    # load params
    state_dict = torch.load(os.path.join(args.output_root, args.checkpoints), map_location=device)
    model.load_state_dict(state_dict)
    # create save folder
    save_folder = os.path.join(args.output_root, 'test_out')
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    with torch.no_grad():
        for mode_label in ['mild', 'moderate', 'severe']:
            per_start_time = time.time()
            test_loader = DataLoader(dataset=test_sets_dict[mode_label], num_workers=args.num_work,
                                     batch_size=args.batch_size, shuffle=False, pin_memory=False)
            print('mode: %s' % mode_label)
            test_ite = 0
            test_psnr = 0
            test_ssim = 0
            eps = 1e-10
            mode_save_folder = os.path.join(save_folder, mode_label)
            if not os.path.exists(mode_save_folder):
                os.makedirs(mode_save_folder)

            for ti, (input, target, names) in enumerate(test_loader):
                realA = input.to(device)
                realB = target.to(device)
                fakeB = model(realA)
                if args.save_image:
                    for fi in range(len(names)):
                        vutils.save_image(fakeB[fi], os.path.join(mode_save_folder, '%s_outputs.png' % names[fi]),
                                          normalize=True)

                # Calculation of SSIM and PSNR values
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
            print('Valid PSNR: {:.4f}'.format(test_psnr))
            print('Valid SSIM: {:.4f}'.format(test_ssim))
            print('test consume time : %f' % (time.time() - per_start_time))
