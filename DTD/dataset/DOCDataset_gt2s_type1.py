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
        logging.info(f'Creating dataset with {len(self.names)} examples')

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        image = cv2.imread(self.imgs_dir + name)
        # image = cv2.resize(image, (512, 512))
        h, w, c = image.shape
        jpg_dct = jpegio.read(self.imgs_dir + name)
        dct_ori = jpg_dct.coef_arrays[0].copy()
        use_qtb2 = jpg_dct.quant_tables[0].copy()

        dct = torch.LongTensor(dct_ori)
        qs = torch.LongTensor(use_qtb2)


        if self.masks_dir is not None:
            mask_name = name.replace('.jpg', '.png')
            mask_name = mask_name.replace('psc_', 'gt3_psc_')
            mask_name = mask_name.replace('ps_', 'gt3_ps_')
            mask_name = mask_name.replace('mosaic_', 'gt3_mosaic_')
            mask_name = mask_name.replace('orig_', 'gt3_orig_')
            mask = Image.open(self.masks_dir + mask_name).convert('L')
            # mask = Image.open(self.masks_dir + name.replace('.jpg', '.png')).convert('L')
            # mask = Image.open(self.masks_dir + name.replace('.tif', '.png')).convert('L')
        else:
            mask = Image.open(self.imgs_dir + name).convert('RGB').convert('L')
        # image = cv2.imread(img_file[0], cv2.IMREAD_COLOR) # 不含alpha通道
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = cv2.imread(img_file[0], cv2.IMREAD_UNCHANGED) # 含alpha通道
        # mask = cv2.imread(mask_file[0], cv2.IMREAD_GRAYSCALE)

        image = np.array(image, np.uint8)
        mask = np.array(mask)
        # print(mask.min(), mask.max())
        # print(image.shape, ocrlabel.shape, mask.shape)

        # 二分类
        # if mask.max() > 1: mask = np.uint8(mask / 255)
        # ---三分类 docimg---
        mask[mask == 255] = 0
        mask[mask == 76] = 1
        mask[mask == 29] = 2
        # # ---二分类 docimg Tamper和Mosaic一类---
        # mask[mask == 255] = 0
        # mask[mask == 76] = 1
        # mask[mask == 29] = 1
        # ---二分类 docimg Orig和Mosaic一类---
        # mask[mask == 255] = 0
        # mask[mask == 76] = 1
        # mask[mask == 29] = 0
        # ---Alinew, SUPATLANTIQUE---
        # mask[mask == 255] = 1

        # print(np.array(image).min(), np.array(image).max()) # 0, 255
        # print(np.array(mask).min(), np.array(mask).max()) # 0, 1



        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        return {
            'image': image,
            'label': mask,
            'dct': np.clip(np.abs(dct), 0, 20),
            'qs': qs,
        }


if __name__ == '__main__':
    from models.dtd import seg_dtd
    data_dir = "/pubdata/lisongze/docimg/exam/docimg2jpeg/test_images_90/"
    labels_dir = "/pubdata/zhengkengtao/docimg/docimg_split811/crop512x512/patch_noblank/test_gt3/"
    img_names = os.listdir(data_dir)[:10]
    # for img in files[:100]:
    #     img_names.append(img.split('.')[0])
    #     img_names.sort()
    img_names.sort()
    print(len(img_names))
    model = seg_dtd('', 2).cuda()
    model = torch.nn.DataParallel(model)

    train_data = DOCDataset(img_names, data_dir, labels_dir, train_transform([512, 512]))
    train_loader = DataLoader(dataset=train_data, batch_size=160, shuffle=False)
    for batch_idx, batch_samples in enumerate(train_loader):
        data, gt, dct, qs = batch_samples['image'].cuda(), batch_samples['label'].cuda(),\
                             batch_samples['dct'].cuda(), batch_samples['qs'].cuda()
        # dct = torch.abs(dct).clamp(0, 20)
        B, C, H, W = data.shape
        qs = qs.reshape(B, 8, 8)
        # gt = gt.to(torch.int64) # [b,h,w]
        gt = gt.to(torch.float16)  # [b,h,w]
        print(data,gt,dct,qs)
        pred = model(data).squeeze(1)  # [b,1,h,w]->[b,h,w]
        print(pred)