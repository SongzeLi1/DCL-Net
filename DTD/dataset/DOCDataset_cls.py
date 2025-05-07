import os
import tempfile

import cv2
import time
import copy

import jpegio
import torch
import random
import logging
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision

from PIL import Image
from tqdm import tqdm
from glob import glob
from torch.utils.data import Dataset, DataLoader

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset.transform import train_transform
# from paddleocr import PaddleOCR, draw_ocr, draw_ocr_box_txt
from PIL import Image, ImageDraw, ImageFont


class DOCDataset(Dataset):
    def __init__(self, names, imgs_dir, masks_dir, transform):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.names = names

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        image = cv2.imread(self.imgs_dir + name)
        image = np.array(image, np.uint8)
        h, w, c = image.shape
        jpg_dct = jpegio.read(self.imgs_dir + name)
        dct_ori = jpg_dct.coef_arrays[0].copy()
        use_qtb2 = jpg_dct.quant_tables[0].copy()

        dct = torch.LongTensor(dct_ori)
        qs = torch.LongTensor(use_qtb2)

        if 'orig' in name:
            clsgt = 0
            gt3 = np.zeros((h, w), np.uint8)
        else:
            gt3name = name.replace('.jpg', '.png')
            gt3name = gt3name.replace('psc_', 'gt3_psc_')
            gt3name = gt3name.replace('ps_', 'gt3_ps_')
            gt3name = gt3name.replace('mosaic_', 'gt3_mosaic_')
            # gt3name = name.replace('_images', '_gt3')
            # gt3name = gt3name.replace('psc_', 'gt3_')
            # gt3name = gt3name.replace('jpg', 'png')
            gt3 = Image.open(self.masks_dir + gt3name).convert('L')
            gt3 = np.array(gt3, np.uint8)
            gt3[gt3 == 255] = 0
            gt3[gt3 == 76] = 1
            gt3[gt3 == 29] = 2
            if 1 in gt3:
                clsgt = 1
            else:
                clsgt = 0


        if self.transform:
            transformed = self.transform(image=image, mask=gt3)
            image = transformed['image']
            mask = transformed['mask']
        return {
            'image': image,
            'label': gt3,
            'clsgt': clsgt,
            'dct': np.clip(np.abs(dct), 0, 20),
            'qs': qs,
        }


if __name__ == '__main__':
    from models.dtd import seg_dtd
    train_names = glob('/pubdata/lisongze/docimg/exam/docimg2jpeg/train_images_75_100/*.jpg')
    train_names.sort()
    # for img in files[:100]:
    #     img_names.append(img.split('.')[0])
    #     img_names.sort()
    print(len(train_names))
    model = seg_dtd('', 2).cuda()
    model = torch.nn.DataParallel(model)

    train_data = DOCDataset(train_names, train_transform([512, 512]))
    train_loader = DataLoader(dataset=train_data, batch_size=160, shuffle=False)
    for batch_idx, batch_samples in enumerate(train_loader):
        data, gt, dct, qs, clsgt = batch_samples['image'].cuda(), batch_samples['label'].cuda(),\
                             batch_samples['dct'].cuda(), batch_samples['qs'].cuda(), batch_samples['clsgt'].cuda()
        # dct = torch.abs(dct).clamp(0, 20)
        B, C, H, W = data.shape
        qs = qs.reshape(B, 8, 8)
        # gt = gt.to(torch.int64) # [b,h,w]
        gt = gt.to(torch.float16)  # [b,h,w]
        print(data,gt,dct,qs,clsgt)
        pred = model(data).squeeze(1)  # [b,1,h,w]->[b,h,w]
        print(pred)