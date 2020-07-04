import os

import cv2
import glob2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.models as models
from torch import nn
from torchvision import transforms


class YourImageData(data.Dataset):
    def __init__(self, root, input='', label=''):
        self.image_folder = root
        # # test data
        self.test_in = os.path.join(self.image_folder, input)
        self.test_gt = os.path.join(self.image_folder, label)
        self.list_in = sorted(glob2.glob(os.path.join(self.test_in, '*.png')))
        self.list_gt = sorted(glob2.glob(os.path.join(self.test_gt, '*.png')))
        self.name_list = [os.path.splitext(os.path.basename(file))[0] for file in self.list_in]
        self.data_len = len(self.list_gt)

    def __getitem__(self, index):
        img_in = cv2.imread(self.list_in[index]) / 255.
        img_gt = cv2.imread(self.list_gt[index]) / 255.

        img_in = img_in[:, :, ::-1] - np.zeros_like(img_in)
        img_gt = img_gt[:, :, ::-1] - np.zeros_like(img_gt)

        img_gt = img_gt.astype(np.float32).transpose(2, 0, 1)
        img_in = img_in.astype(np.float32).transpose(2, 0, 1)

        img_gt = torch.from_numpy(img_gt)
        img_in = torch.from_numpy(img_in)
        base_name = self.name_list[index]
        return img_in, img_gt, base_name

    def __len__(self):
        return self.data_len


class Perceptual_vgg_loss(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Perceptual_vgg_loss, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.l1_loss = nn.L1Loss()
        for x in range(1):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])

        for x in range(1, 6):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x1, x2):
        h1_1 = self.slice1(x1)
        h1_2 = self.slice1(x2)
        h2_1 = self.slice2(h1_1)
        h2_2 = self.slice2(h1_2)
        loss = self.l1_loss(h1_1, h1_2) + self.l1_loss(h2_1, h2_2)
        return loss


def save_networks(model, save_path):
    torch.save(model.state_dict(), save_path)
    print('save model {}'.format(save_path))


def load_imgs(list_in, list_gt, size=63):
    assert len(list_in) == len(list_gt)
    img_num = len(list_in)
    imgs_in = np.zeros([img_num, size, size, 3])
    imgs_gt = np.zeros([img_num, size, size, 3])
    for k in range(img_num):
        imgs_in[k, ...] = cv2.imread(list_in[k]) / 255.
        imgs_gt[k, ...] = cv2.imread(list_gt[k]) / 255.
    return imgs_in, imgs_gt


def data_reformat(data):
    """RGB <--> BGR, swap H and W"""
    assert data.ndim == 4
    out = data[:, :, :, ::-1] - np.zeros_like(data)
    out = np.swapaxes(out, 1, 2)
    out = out.astype(np.float32)
    return out


class NpzData(data.Dataset):
    def __init__(self, root):
        self.train_dir = root
        # training data
        self.paths = sorted(glob2.glob(os.path.join(self.train_dir, '*.npz')))
        self.data_len = len(self.paths)

    def __getitem__(self, index):
        data = np.load(self.paths[index])
        img = data['input']
        img_gt = data['label']

        img = torch.from_numpy(img)
        img_gt = torch.from_numpy(img_gt)
        return img, img_gt

    def __len__(self):
        return self.data_len


class ImageData(data.Dataset):
    def __init__(self, root, mode='moderate'):
        self.image_folder = root
        # # test data
        self.test_in = os.path.join(self.image_folder, mode + '_in/')
        self.test_gt = os.path.join(self.image_folder, mode + '_gt/')
        list_in = sorted(glob2.glob(os.path.join(self.test_in, '*.png')))
        list_gt = sorted(glob2.glob(os.path.join(self.test_gt, '*.png')))
        self.name_list = [os.path.splitext(os.path.basename(file))[0] for file in list_in]
        self.data_all, self.label_all = load_imgs(list_in, list_gt)
        self.transform = transforms.Compose([transforms.ToTensor()])
        # data reformat, because the data for tools training are in a different format
        self.data = data_reformat(self.data_all)
        self.label = data_reformat(self.label_all)
        self.data_len = len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        img_gt = self.label[index]
        img = self.transform(img)
        img_gt = self.transform(img_gt)
        img_name = self.name_list[index]
        return img, img_gt, img_name

    def __len__(self):
        return self.data_len
